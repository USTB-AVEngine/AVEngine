"""五种呈现的输入清单，以及单声道下混。

这里钉三件事。第一，下混的数值性质：左右完全一样时输出必须等于原信号，否则"去掉双耳线索"这个
对照组里就混进了别的改动。第二，五份输入的题面必须逐字相同——题面一旦被改写，后面比出来的
"掉了几分"就不再只是模态的差别。第三，这一步只新增文件：跑之前跑之后 public 下原有文件的哈希
必须一个不差。
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from avengine.dataset.presentations import (
    MONO_METHODS,
    PRESENTATIONS,
    PresentationError,
    downmix_to_mono,
    file_sha256,
    presentation_rows,
    text_is_identical,
    write_mono_wav,
)

REPOSITORY = Path(__file__).resolve().parents[1]
RATE = 16000


def child_env(home):
    """PYTHONPATH 是两段：本检出的 src，加上原生扩展那一段。

    漏掉第二段的话，子进程会当场报 "No module named soundfile"，看上去像环境坏了，
    其实只是路径少写了一截。这里从当前环境继承，而不是手写一段。
    """

    import os

    inherited = os.environ.get("PYTHONPATH", "")
    entries = [str(REPOSITORY / "src")] + [e for e in inherited.split(os.pathsep) if e]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(entries),
            "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(home)}


def stereo(left, right):
    return np.stack([np.asarray(left, dtype=np.float64),
                     np.asarray(right, dtype=np.float64)], axis=1)


# --------------------------------------------------------------------------
# 下混的数值性质
# --------------------------------------------------------------------------


def test_左右相同时输出就是原信号():
    left = np.array([0.0, 0.25, -0.5, 0.125])
    assert np.array_equal(downmix_to_mono(stereo(left, left)), left)


def test_左右等权平均():
    got = downmix_to_mono(stereo([1.0, 0.0], [0.0, 1.0]))
    assert np.allclose(got, [0.5, 0.5])


def test_下混不会比两路里大的那一路更大():
    left, right = [0.9, -0.8, 0.1], [0.7, -0.6, -0.1]
    got = downmix_to_mono(stereo(left, right))
    assert np.max(np.abs(got)) <= max(np.max(np.abs(left)), np.max(np.abs(right)))


def test_反相的两路会互相抵消而不是报错():
    left = np.array([0.5, -0.5])
    assert np.allclose(downmix_to_mono(stereo(left, -left)), [0.0, 0.0])


def test_双耳平均只收两路():
    with pytest.raises(PresentationError):
        downmix_to_mono(np.zeros((4, 4)), "binaural_mean")


def test_FOA_取第零路原样():
    four = np.stack([np.array([1.0, 2.0]), np.array([9.0, 9.0]),
                     np.zeros(2), np.zeros(2)], axis=1)
    assert np.array_equal(downmix_to_mono(four, "foa_w"), [1.0, 2.0])


def test_不认识的方法直接报错():
    with pytest.raises(PresentationError):
        downmix_to_mono(np.zeros((2, 2)), "average_of_everything")
    assert set(MONO_METHODS) == {"binaural_mean", "foa_w"}


def test_落盘的单声道带得出来源收据(tmp_path):
    source = tmp_path / "mixture.wav"
    left = np.linspace(-0.4, 0.4, 100)
    sf.write(str(source), stereo(left, left), RATE, subtype="FLOAT")
    destination = tmp_path / "mono.wav"
    receipt = write_mono_wav(source, destination, method="binaural_mean")

    assert receipt["channels_in"] == 2 and receipt["channels_out"] == 1
    assert receipt["sha256"] == file_sha256(destination)
    assert receipt["derived_from_sha256"] == file_sha256(source)
    assert receipt["sample_rate_hz"] == RATE and receipt["frames"] == 100
    assert receipt["normalisation"] == "none"
    assert receipt["clipped_samples"] == 0
    assert receipt["peak_after"] <= receipt["peak_before"] + 1e-12
    # 左右相同，所以读回来就是原信号
    written, rate = sf.read(str(destination), dtype="float64")
    assert rate == RATE
    assert np.array_equal(written, np.asarray(left, dtype=np.float32).astype(np.float64))


def test_子类型跟着源文件走(tmp_path):
    source = tmp_path / "s.wav"
    sf.write(str(source), stereo([0.5, 0.25], [0.5, 0.25]), RATE, subtype="PCM_16")
    receipt = write_mono_wav(source, tmp_path / "m.wav", method="binaural_mean")
    assert receipt["subtype"] == "PCM_16"
    assert sf.info(str(tmp_path / "m.wav")).subtype == "PCM_16"


# --------------------------------------------------------------------------
# 输入清单
# --------------------------------------------------------------------------


QUESTION = {
    "question_id": "question_000001",
    "qa_id": "QA-04",
    "forms": {"open": {"question_en": "left or right?"},
              "mcq": {"question_en": "left or right?",
                      "options": [{"option": "A", "label_en": "left"},
                                  {"option": "B", "label_en": "right"}]}},
    "required_modalities": None,
    "media": {"video": "media/video_abc.mp4", "audio": "media/audio_abc.wav"},
    "media_clock": {"frame_count": 150, "frame_rate_hz": 15},
}


def test_五种呈现各拿到该拿的媒体():
    rows = presentation_rows(QUESTION, mono_audio="media/audio_mono_abc.wav")
    assert [row["presentation"] for row in rows] == list(PRESENTATIONS)
    got = {row["presentation"]: row["media"] for row in rows}
    assert set(got["av"]) == {"video", "audio"}
    assert set(got["a_only"]) == {"audio"}
    assert set(got["v_only"]) == {"video"}
    assert got["t_only"] == {}
    assert got["av_mono"]["audio"] == "media/audio_mono_abc.wav"
    assert got["av_mono"]["video"] == QUESTION["media"]["video"]
    assert got["av"]["audio"] != got["av_mono"]["audio"]


def test_五份输入共用一个_question_id_题面逐字相同():
    rows = presentation_rows(QUESTION, mono_audio="media/audio_mono_abc.wav")
    assert {row["question_id"] for row in rows} == {"question_000001"}
    for row in rows:
        assert row["forms"] == QUESTION["forms"]
    assert text_is_identical(rows)["identical"] is True


def test_单声道那一行声道数记成一():
    rows = {r["presentation"]: r for r in presentation_rows(QUESTION, mono_audio="m.wav")}
    assert rows["av_mono"]["audio_channels"] == 1
    assert rows["av"]["audio_channels"] == 2
    assert rows["t_only"]["audio_channels"] is None


def test_题面被改写会被抓出来():
    rows = presentation_rows(QUESTION, mono_audio="m.wav")
    rows[1] = {**rows[1], "forms": {"open": {"question_en": "which side?"}}}
    checked = text_is_identical(rows)
    assert checked["identical"] is False
    assert checked["questions_with_differing_text"] == ["question_000001"]


def test_没有音频就不要假装拼得出来():
    with pytest.raises(PresentationError):
        presentation_rows({**QUESTION, "media": {"video": "v.mp4"}}, mono_audio=None)


# --------------------------------------------------------------------------
# 端到端：只新增，不改动
# --------------------------------------------------------------------------


def tiny_bank(root: Path):
    (root / "public").mkdir(parents=True)
    (root / "media").mkdir()
    left = np.linspace(-0.3, 0.3, 64)
    sf.write(str(root / "media" / "audio_abc.wav"), stereo(left, -left), RATE, subtype="FLOAT")
    (root / "media" / "video_abc.mp4").write_bytes(b"not really a video")
    (root / "public" / "questions.jsonl").write_text(
        json.dumps(QUESTION, ensure_ascii=False) + "\n")
    (root / "public" / "other_public_file.json").write_text('{"kept": true}\n')
    return root


def test_跑完之后原有公开文件逐字节没变(tmp_path):
    bank = tiny_bank(tmp_path / "bank_run")
    before = {p.name: file_sha256(p) for p in (bank / "public").iterdir()}

    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "tools" / "dataset" / "export_presentations.py"), str(bank)],
        capture_output=True, text=True, cwd=str(REPOSITORY),
        env=child_env(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    after = {p.name: file_sha256(p) for p in (bank / "public").iterdir()}
    for name, digest in before.items():
        assert after[name] == digest, f"{name} 被改了"
    assert set(after) - set(before) == {
        "model_inputs_by_presentation.jsonl", "presentation_media.jsonl"}

    rows = [json.loads(l) for l in (bank / "public" / "model_inputs_by_presentation.jsonl").open()]
    assert len(rows) == len(PRESENTATIONS)
    receipts = [json.loads(l) for l in (bank / "public" / "presentation_media.jsonl").open()]
    assert len(receipts) == 1 and receipts[0]["method"] == "binaural_mean"
    report = json.loads((bank / "presentations_report.json").read_text())
    assert report["public_files_unchanged"] is True
    assert report["text_identity"]["identical"] is True
    assert report["rows"] == len(PRESENTATIONS)


def test_不覆盖已经存在的清单(tmp_path):
    bank = tiny_bank(tmp_path / "bank_run")
    (bank / "public" / "model_inputs_by_presentation.jsonl").write_text("{}\n")
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "tools" / "dataset" / "export_presentations.py"), str(bank)],
        capture_output=True, text=True, cwd=str(REPOSITORY),
        env=child_env(tmp_path),
    )
    assert result.returncode != 0
    assert "只新增不覆盖" in (result.stdout + result.stderr)
