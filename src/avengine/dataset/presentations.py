"""一道题的五种呈现，以及单声道下混。

评测想回答「去掉一个模态还剩多少」，就得让同一道题以几种不同的输入喂给模型：两个都给、只给音频、
只给视频、连媒体都不给、以及给视频加一条我们自己下混的单声道。最后那一种是专门用来看空间线索的：
画面还在、声音还在，只有左右耳的差别没了，掉下去的分就是双耳线索撑着的那一部分。

三条硬规矩写在这里，因为它们最容易在别处被悄悄破坏：

* **题面一个字都不能变。** 五份输入共用一个 ``question_id``，只多一个 ``presentation`` 字段；
  题干、选项、选项顺序必须逐字相同，否则比出来的差异里混着措辞的影响，这个实验就白做了。
* **单声道必须是我们自己下混的，并且带收据。** 哪个文件、用什么方法、下混前后的峰值各是多少，
  都要能回查，不能只留一个 wav 让人猜它是怎么来的。
* **只新增，不改动。** 这些是新文件，现有的公开文件一个字节都不动。

这个模块只管「怎么算」，不管题库长什么样；按题库目录落盘的是
``tools/dataset/export_presentations.py``，成组的那一路（WP-G）也接这里。
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

#: 五种呈现，顺序就是导出时的行顺序。
PRESENTATIONS: tuple[str, ...] = ("av", "a_only", "v_only", "t_only", "av_mono")

PRESENTATION_NOTES: dict[str, str] = {
    "av": "视频加空间音频，也就是现在那一份输入。",
    "a_only": "只给音频和题面，不给视频。",
    "v_only": "只给视频和题面，不给音频。",
    "t_only": "只给题面，媒体一概不给；它量的是光靠先验能蒙对多少。",
    "av_mono": "视频加我们自己下混的单声道。画面和声音都还在，只有双耳差没了。",
}

#: 单声道怎么来。默认双耳等权平均。
MONO_METHODS: tuple[str, ...] = ("binaural_mean", "foa_w")

MONO_METHOD_NOTES: dict[str, str] = {
    "binaural_mean": (
        "双耳两路等权平均 (L+R)/2。左右完全一样时输出等于原信号，"
        "并且平均值的绝对值不会超过两路里大的那一路，所以不会削顶。"
    ),
    "foa_w": (
        "取 FOA 的第 0 路（W，全向分量）原样输出。它是另一条独立的单声道来源，"
        "但幅度取决于 FOA 的归一化约定（SN3D / N3D 等），跟双耳平均不在同一个刻度上，"
        "所以两种方法的绝对电平不能直接比，只能各自比。"
    ),
}

#: 哪种呈现给哪些媒体。
PRESENTATION_MEDIA_KINDS: dict[str, tuple[str, ...]] = {
    "av": ("video", "audio"),
    "a_only": ("audio",),
    "v_only": ("video",),
    "t_only": (),
    "av_mono": ("video", "audio"),
}


class PresentationError(ValueError):
    """呈现变体生成不下去了，并且说得出为什么。"""


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def downmix_to_mono(samples: np.ndarray, method: str = "binaural_mean") -> np.ndarray:
    """把多声道下混成一路。

    ``binaural_mean`` 要求正好两路，输出 (L+R)/2；``foa_w`` 取第 0 路原样。
    两种都不做归一化、不做限幅——归一化会把不同题之间的响度关系改掉，那是另一件事。
    """

    if method not in MONO_METHODS:
        raise PresentationError(f"不认识的单声道方法 {method!r}，只有 {list(MONO_METHODS)}")
    array = np.asarray(samples)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0:
        raise PresentationError("下混要的是形如 (帧数, 声道数) 的样本")
    if method == "binaural_mean":
        if array.shape[1] != 2:
            raise PresentationError(f"双耳平均要正好两路，拿到的是 {array.shape[1]} 路")
        return (array[:, 0] + array[:, 1]) / 2.0
    if array.shape[1] < 1:
        raise PresentationError("FOA 取 W 路至少要有一路")
    return array[:, 0].copy()


def _levels(samples: np.ndarray) -> dict[str, float]:
    array = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0
    return {"peak": peak, "rms": rms}


def write_mono_wav(
    source_path: str | Path,
    destination: str | Path,
    *,
    method: str = "binaural_mean",
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """下混一个文件并落盘，返回这份新媒体的收据。

    收据要能让人不看代码就回答「这个 wav 是哪来的」：源文件、源文件的哈希、方法、
    下混前后的峰值和有效值、输出自己的哈希和字节数。
    """

    source_path = Path(source_path)
    destination = Path(destination)
    samples, rate = sf.read(str(source_path), dtype="float64", always_2d=True)
    info = sf.info(str(source_path))
    mono = downmix_to_mono(samples, method)
    before, after = _levels(samples), _levels(mono)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + f".tmp_{os.getpid()}")
    # 临时名没有 .wav 后缀，格式必须显式给，否则 soundfile 猜不出来。容器和子类型都跟源文件一致。
    sf.write(str(temp), mono, int(rate), subtype=info.subtype, format=info.format)
    os.replace(temp, destination)
    return {
        "path": str(destination),
        "sha256": file_sha256(destination),
        "bytes": destination.stat().st_size,
        "frames": int(mono.shape[0]),
        "sample_rate_hz": int(rate),
        "channels_in": int(samples.shape[1]),
        "channels_out": 1,
        "subtype": info.subtype,
        "method": method,
        "method_note": MONO_METHOD_NOTES[method],
        "derived_from": str(source_path),
        "derived_from_sha256": source_sha256 or file_sha256(source_path),
        "peak_before": before["peak"],
        "peak_after": after["peak"],
        "rms_before": before["rms"],
        "rms_after": after["rms"],
        "normalisation": "none",
        "clipped_samples": int(np.sum(np.abs(mono) > 1.0)),
    }


def presentation_rows(
    question: Mapping[str, Any],
    *,
    mono_audio: str | None,
    presentations: Sequence[str] = PRESENTATIONS,
) -> list[dict[str, Any]]:
    """一道题的五行输入清单。题面原样抄过去，只换媒体。"""

    media = question.get("media") or {}
    video, audio = media.get("video"), media.get("audio")
    rows = []
    for presentation in presentations:
        if presentation not in PRESENTATION_MEDIA_KINDS:
            raise PresentationError(f"不认识的呈现 {presentation!r}")
        kinds = PRESENTATION_MEDIA_KINDS[presentation]
        row_media: dict[str, Any] = {}
        if "video" in kinds:
            if not video:
                raise PresentationError(f"{presentation} 要视频，可这道题没有")
            row_media["video"] = video
        if "audio" in kinds:
            wanted = mono_audio if presentation == "av_mono" else audio
            if not wanted:
                raise PresentationError(f"{presentation} 要音频，可这道题没有")
            row_media["audio"] = wanted
        row = {
            "question_id": question["question_id"],
            "qa_id": question["qa_id"],
            "presentation": presentation,
            "presentation_note": PRESENTATION_NOTES[presentation],
            # 逐字照抄，连选项顺序都不动。
            "forms": question["forms"],
            "required_modalities": question.get("required_modalities"),
            "media": row_media,
            "audio_channels": (1 if presentation == "av_mono"
                               else (2 if "audio" in kinds else None)),
        }
        if question.get("media_clock") is not None:
            row["media_clock"] = question["media_clock"]
        rows.append(row)
    return rows


def text_is_identical(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """同一道题的几份输入，题面和选项是不是逐字一样。

    这条必须是可查的数字而不是一句承诺：题面一旦在某个呈现里被改写，后面比出来的
    「掉了几分」就不再只是模态的差别。
    """

    import json

    by_question: dict[str, list[str]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], []).append(
            json.dumps(row["forms"], ensure_ascii=False, sort_keys=True)
        )
    differing = [qid for qid, texts in by_question.items() if len(set(texts)) != 1]
    return {
        "questions_checked": len(by_question),
        "rows_checked": len(rows),
        "questions_with_differing_text": differing,
        "identical": not differing,
    }
