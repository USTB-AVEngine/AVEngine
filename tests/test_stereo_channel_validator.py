"""Unit tests for the stereo-channel validator (pilot item 1.1).

阳性对照规矩:检查器宣布"通过"之前,先证明它在已知坏对象上会失败——
本测试集内置四类坏对象(单声道、L==R 复制、双声道静音、空文件由读取层
拒绝),全部必须 FAIL;真立体声必须 PASS。全部样本由测试自行合成,
不依赖外部数据。
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from validate_stereo_channels import main, validate_wav  # noqa: E402


def _write_wav(path: Path, left: np.ndarray, right: np.ndarray | None, rate: int = 16000):
    """写 int16 wav;right=None 时写单声道。"""
    if right is None:
        data = np.asarray(left)
        channels = 1
    else:
        data = np.stack([left, right], axis=1)
        channels = 2
    pcm = np.clip(np.asarray(data) * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


@pytest.fixture()
def tone():
    t = np.arange(16000) / 16000.0
    return np.sin(2 * np.pi * 440 * t) * 0.5


def test_true_stereo_passes(tmp_path, tone):
    right = np.roll(tone, 37) * 0.8  # 相位与幅度都不同的右声道
    p = tmp_path / "stereo.wav"
    _write_wav(p, tone, right)
    rec = validate_wav(str(p))
    assert rec["status"] == "pass"
    assert rec["channels"] == 2
    assert rec["diff_ratio"] > 0.5


def test_positive_control_mono_fails(tmp_path, tone):
    p = tmp_path / "mono.wav"
    _write_wav(p, tone, None)
    rec = validate_wav(str(p))
    assert rec["status"] == "fail"
    assert "2 channels" in rec["reason"]


def test_positive_control_duplicated_channels_fail(tmp_path, tone):
    p = tmp_path / "dup.wav"
    _write_wav(p, tone, tone.copy())  # 历史事故的等价形态:双声道但内容相同
    rec = validate_wav(str(p))
    assert rec["status"] == "fail"
    assert "nearly identical" in rec["reason"]


def test_positive_control_silent_stereo_fails(tmp_path):
    z = np.zeros(16000)
    p = tmp_path / "silent.wav"
    _write_wav(p, z, z)
    rec = validate_wav(str(p))
    assert rec["status"] == "fail"


def test_threshold_is_explicit_parameter(tmp_path, tone):
    # 只有 0.5% 的样本不同:默认阈值(1%)应拒,放宽到 0.1% 应过
    right = tone.copy()
    n = tone.size
    idx = np.arange(0, n, 200)  # 0.5%
    right[idx] += 0.2
    p = tmp_path / "sparse.wav"
    _write_wav(p, tone, right)
    assert validate_wav(str(p))["status"] == "fail"
    assert validate_wav(str(p), min_diff_ratio=0.001)["status"] == "pass"


def test_cli_fails_fast_and_manifest_no_clobber(tmp_path, tone):
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    _write_wav(good, tone, np.roll(tone, 41))
    _write_wav(bad, tone, tone.copy())
    manifest = tmp_path / "m.json"
    # 目录模式:含一个坏文件 → 非零退出
    assert main([str(tmp_path), "--manifest", str(manifest)]) == 1
    assert manifest.exists()
    # no-clobber:同名 manifest 再跑必须拒绝
    assert main([str(good), "--manifest", str(manifest)]) == 2


def test_cli_all_good_returns_zero(tmp_path, tone):
    good = tmp_path / "good.wav"
    _write_wav(good, tone, np.roll(tone, 41))
    assert main([str(good)]) == 0
