"""Unit tests for the physical-feature probe (pilot item 1.6).

阳性对照的双向要求:①对**有**已知物理侧信道(响度差、双耳强度差、
双耳延迟、事件密度)的合成数据,探针必须能分对(否则探针失明,认证
会错误放行);②对**打乱标签**的同一批数据,探针必须掉回随机线附近
(否则探针虚报,认证会错误降级)。另含 oracle 条款的静态自检:模块
源码不得触碰任何引擎侧真值。
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

import probe_physical_features as ppf  # noqa: E402
from probe_physical_features import FEATURE_NAMES, extract_features, main, probe  # noqa: E402

RATE = 16000


def _write_wav(path: Path, left: np.ndarray, right: np.ndarray):
    pcm = np.clip(np.stack([left, right], axis=1) * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as h:
        h.setnchannels(2)
        h.setsampwidth(2)
        h.setframerate(RATE)
        h.writeframes(pcm.tobytes())


def _noise(rng, n=RATE):
    return rng.standard_normal(n) * 0.1


def _make_ild_pair(tmp, rng, i, side):
    base = _noise(rng)
    gain_l, gain_r = (1.0, 0.3) if side == "left" else (0.3, 1.0)
    p = tmp / f"ild_{side}_{i}.wav"
    _write_wav(p, base * gain_l, base * gain_r)
    return p


def _rng():
    return np.random.default_rng(7)


def test_ild_side_channel_is_caught_and_shuffled_labels_are_not(tmp_path):
    rng = _rng()
    feats, labels = [], []
    for i in range(12):
        for side in ("left", "right"):
            p = _make_ild_pair(tmp_path, rng, i, side)
            f = extract_features(str(p))
            feats.append([f[k] for k in FEATURE_NAMES])
            labels.append(side)
    x = np.array(feats)
    acc, _ = probe(x, labels, folds=4, seed="s")
    assert acc >= 0.9  # 阳性:真实侧信道必须被抓
    # 反向对照测的是期望性质:单次打乱在 24 样本上方差太大(σ≈0.10,
    # 初版单种子断言 0.75 被 +2.9σ 的一次打乱击穿),改为 20 次打乱取均值,
    # 均值必须贴随机线;探针取双分类器较高者带小样本正偏,方向是安全的
    # (宁可误降级不可误放行),均值上限放 0.65。
    shuffle_accs = []
    for k in range(20):
        shuffled = list(labels)
        np.random.default_rng(k).shuffle(shuffled)
        acc_shuf, _ = probe(x, shuffled, folds=4, seed="s")
        shuffle_accs.append(acc_shuf)
    assert float(np.mean(shuffle_accs)) <= 0.65
    assert max(shuffle_accs) < acc  # 任何一次打乱都不得追平真信号


def test_rms_trend_side_channel(tmp_path):
    rng = _rng()
    feats, labels = [], []
    ramp_up = np.linspace(0.2, 1.0, RATE)
    for i in range(10):
        base = _noise(rng)
        for lab, env in (("approach", ramp_up), ("recede", ramp_up[::-1])):
            p = tmp_path / f"rms_{lab}_{i}.wav"
            _write_wav(p, base * env, base * env)
            f = extract_features(str(p))
            feats.append([f[k] for k in FEATURE_NAMES])
            labels.append(lab)
    acc, _ = probe(np.array(feats), labels, folds=4, seed="s")
    assert acc >= 0.9


def test_itd_side_channel(tmp_path):
    rng = _rng()
    feats, labels = [], []
    for i in range(10):
        base = _noise(rng)
        for lab, lag in (("left_lead", 8), ("right_lead", -8)):  # ±0.5ms
            p = tmp_path / f"itd_{lab}_{i}.wav"
            _write_wav(p, base, np.roll(base, lag))
            f = extract_features(str(p))
            feats.append([f[k] for k in FEATURE_NAMES])
            labels.append(lab)
    acc, _ = probe(np.array(feats), labels, folds=4, seed="s")
    assert acc >= 0.9


def test_event_density_feature(tmp_path):
    rng = _rng()
    silent = np.zeros(RATE)

    def bursts(k):
        x = silent.copy()
        for j in range(k):
            start = int((j + 0.5) / k * RATE)
            x[start:start + 800] = rng.standard_normal(800) * 0.5
        return x

    one, three = bursts(1), bursts(3)
    p1, p3 = tmp_path / "e1.wav", tmp_path / "e3.wav"
    _write_wav(p1, one, one)
    _write_wav(p3, three, three)
    f1, f3 = extract_features(str(p1)), extract_features(str(p3))
    assert f3["event_density"] > f1["event_density"]


def test_probe_requires_stereo(tmp_path):
    mono = _noise(_rng())
    pcm = np.clip(mono * 32767.0, -32768, 32767).astype("<i2")
    p = tmp_path / "mono.wav"
    with wave.open(str(p), "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(RATE)
        h.writeframes(pcm.tobytes())
    with pytest.raises(ValueError):
        extract_features(str(p))


def test_oracle_clause_static_self_check():
    src = Path(ppf.__file__).read_text(encoding="utf-8")
    body = src.split('"""', 2)[2]  # 跳过模块 docstring,只查代码体
    for forbidden in ("timeline", "audio_program", "source_asset_runtime",
                      "fact_table", "registry", "spec.json", "rir", "dry"):
        assert forbidden not in body.lower(), f"oracle clause violated: {forbidden!r} in code"


def test_deterministic_across_runs(tmp_path):
    rng = _rng()
    feats, labels = [], []
    for i in range(8):
        for side in ("left", "right"):
            p = _make_ild_pair(tmp_path, rng, 100 + i, side)
            f = extract_features(str(p))
            feats.append([f[k] for k in FEATURE_NAMES])
            labels.append(side)
    x = np.array(feats)
    a1 = probe(x, labels, folds=4, seed="fixed")
    a2 = probe(x, labels, folds=4, seed="fixed")
    assert a1[0] == a2[0] and a1[1] == a2[1]


def test_cli_end_to_end_no_clobber_and_fail_fast(tmp_path):
    rng = _rng()
    items = []
    for i in range(6):
        for side in ("left", "right"):
            p = _make_ild_pair(tmp_path, rng, 200 + i, side)
            items.append({"question_id": f"q_{side}_{i}", "wav": str(p),
                          "label": side, "group_id": f"g{i}"})
    items_p = tmp_path / "items.json"
    items_p.write_text(json.dumps(items))
    out = tmp_path / "probe.json"
    assert main(["--items", str(items_p), "--folds", "3", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["accuracy_all_features"] >= 0.9
    assert doc["predictions"][0]["group_id"] == "g0"
    assert main(["--items", str(items_p), "--out", str(out)]) == 2  # no-clobber
    # 坏 wav → 非零退出(失败即停)
    items.append({"question_id": "broken", "wav": str(tmp_path / "nope.wav"), "label": "left"})
    items_p.write_text(json.dumps(items))
    assert main(["--items", str(items_p), "--out", str(tmp_path / "p2.json")]) == 1
