"""The listening copy raises a pack to an audible level without touching the render."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from build_qa_v3_listening_copy import (  # noqa: E402
    CEILING_TRUE_PEAK_DBTP,
    ListeningCopyError,
    build_listening_copy,
    listening_gain_db,
)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def test_the_gain_puts_the_loudest_clip_on_the_target():
    # 实测这 18 段的真峰是 -37.94 到 -31.60 dBTP
    assert listening_gain_db([-31.60, -33.69, -37.94]) == pytest.approx(28.60)
    # 只由最响那一条决定，加进更轻的条目不改变结果
    assert listening_gain_db([-31.60, -60.0]) == pytest.approx(28.60)


def test_a_target_that_would_clip_fails_closed():
    # 削波会静默毁掉 ILD,而 ILD 正是这批题要考的线索
    with pytest.raises(ListeningCopyError, match="above"):
        listening_gain_db([-31.60], target=0.0)


def test_no_measurement_means_no_guessed_level():
    with pytest.raises(ListeningCopyError, match="nothing to level"):
        listening_gain_db([])


def _pack(root, names=("a", "b")):
    pub = root / "public" / "media"
    pub.mkdir(parents=True)
    (root / "public" / "study_items.json").write_text(json.dumps(
        {"schema": "qa_v3_human_calibration_study_v1", "items": []}))
    (root / "public" / "index.html").write_text("<!doctype html>")
    for name in names:
        _tone(pub / f"{name}.mp4", 0.02 if name == "a" else 0.2)
    return root


def _tone(path, amplitude):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"sine=frequency=440:duration=1:sample_rate=16000",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1:r=15",
         "-map", "1:v", "-map", "0:a", "-af", f"volume={amplitude}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)], check=True, capture_output=True)


def test_it_refuses_to_write_over_the_pack(tmp_path):
    pack = _pack(tmp_path / "pack") if HAVE_FFMPEG else (tmp_path / "pack")
    if not HAVE_FFMPEG:
        (pack / "public" / "media").mkdir(parents=True)
    with pytest.raises(ListeningCopyError, match="over the pack"):
        build_listening_copy(pack, pack)


def test_it_refuses_a_non_empty_output(tmp_path):
    pack = tmp_path / "pack"
    (pack / "public" / "media").mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    (out / "already-here").write_text("x")
    with pytest.raises(FileExistsError, match="non-empty"):
        build_listening_copy(pack, out)


def test_it_refuses_something_that_is_not_a_pack(tmp_path):
    empty = tmp_path / "nope"
    empty.mkdir()
    with pytest.raises(ListeningCopyError, match="not a built calibration pack"):
        build_listening_copy(empty, tmp_path / "out")


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg is needed to measure a level")
def test_one_scalar_for_the_pack_keeps_the_level_difference(tmp_path):
    pack = _pack(tmp_path / "pack")
    manifest = build_listening_copy(pack, tmp_path / "out")
    rows = manifest["clips"]
    assert len(rows) == 2
    gain = manifest["listening_gain_db"]
    # 最响那一条落在靶心
    peaks = [row["output_true_peak_dbtp"] for row in rows.values()]
    assert max(peaks) <= CEILING_TRUE_PEAK_DBTP
    assert max(peaks) == pytest.approx(-3.0, abs=0.6)
    # 整包同一个标量:两条之间的电平差保持不变
    before = (rows["b.mp4"]["input_true_peak_dbtp"]
              - rows["a.mp4"]["input_true_peak_dbtp"])
    after = (rows["b.mp4"]["output_true_peak_dbtp"]
             - rows["a.mp4"]["output_true_peak_dbtp"])
    assert after == pytest.approx(before, abs=0.6)
    # 原始包一个字节没动,新树自己带清单与规则
    assert (pack / "public/media/a.mp4").is_file()
    study = json.loads((tmp_path / "out/public/study_items.json").read_text())
    assert study["listening_gain_db"] == gain
    assert "max(input_tp" in study["listening_gain_rule"]
    assert (tmp_path / "out/listening_copy_manifest.json").is_file()
