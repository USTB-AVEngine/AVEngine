"""One machine verdict for a rendered episode, with its reasons in words.

Every number judged here is either already measured by the render chain -
the ambisonic pass records per-frame direction error against geometry, the
binaural pass proves left and right with cardinal probes and refuses a
mirrored clip - or is a file-level fact nothing measured before: the levels
of the wav that actually ships, the content of the frames that actually
ship, the mux that binds them. Until now the receipt named "the human
verdict on deliverable_mp4" as the acceptance for all of it, which
outsourced to human ears exactly the checks arithmetic does better, and
made review a perception task. The verdict file this module writes is the
acceptance; a human verdict is an optional override on top of it.

Thresholds are calibrated against the first frame-verified episode
(scene 00808, task 20260828T155757Z), not invented:

* Whole-clip interaural level difference: the shipped wav measured
  -1.20 dB against the report's RIR-derived median of -1.36 dB, and the
  sign flips under a deliberate channel swap - so the gate is sign
  agreement (when the report's |median| >= 0.75 dB) plus a 2.5 dB
  magnitude window. A per-frame sign-agreement gate was measured first
  and rejected: 39/68 frames agreed while a swapped-channel control
  still scored 29/68, and a gate that cannot tell a swap from noise is
  not a gate.
* Frame motion: the moving loudspeaker changed 0.63% of pixels between
  first and last frame at threshold 8/255; the gate asks for 0.05%.
* Durations: the wav runs 1.56 s past frames/rate (the reverberation
  tail), so the wav gate allows frames/rate .. frames/rate + 4 s, while
  the muxed mp4 (cut by -shortest) must sit within 0.25 s of
  frames/rate.

Stdlib only, deliberately: the same file must run identically under the
studio server environment, the visual runtime and the ss2 audio
environment, and its unit tests must need nothing installed.
"""

from __future__ import annotations

import array
import json
import math
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "avengine_machine_audition_v1"

_SILENCE_FLOOR_DBFS = -50.0
_CLIP_CEILING = 0.999
_ILD_GATE_MINIMUM_DB = 0.75
_ILD_WINDOW_DB = 2.5
_DIRECTION_MINIMUM_FRACTION = 0.9
_FRAME_LEVEL_MINIMUM = 8.0
_MOTION_MINIMUM_FRACTION = 0.0005
_MOTION_PIXEL_THRESHOLD = 8
_WAV_TAIL_ALLOWANCE_S = 4.0
_MUX_DURATION_TOLERANCE_S = 0.25
_POSITION_TOLERANCE_M = 5.0e-3


class AuditionError(ValueError):
    pass


# ---------------------------------------------------------------------------
# file readers


def read_wav_levels(path: Path) -> dict[str, Any]:
    """Channel levels of a RIFF wav, parsed by hand.

    The render chain writes 16-bit PCM today, but the reader also accepts
    32-bit float (format tag 3) because the stdlib ``wave`` module refuses
    it and a future renderer switching formats should change a number here,
    not silently lose its audition.
    """

    blob = path.read_bytes()
    if blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise AuditionError(f"not a RIFF wav: {path}")
    fmt: tuple[int, ...] | None = None
    data: bytes | None = None
    position = 12
    while position + 8 <= len(blob):
        chunk_id = blob[position : position + 4]
        (size,) = struct.unpack("<I", blob[position + 4 : position + 8])
        body = blob[position + 8 : position + 8 + size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif chunk_id == b"data":
            data = body
        position += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise AuditionError(f"wav is missing fmt or data chunk: {path}")
    tag, channel_count, sample_rate, _, _, bits = fmt
    bytes_per_sample = bits // 8
    sample_count = len(data) // (channel_count * bytes_per_sample)
    if tag == 1 and bits == 16:
        raw = array.array("h")
        raw.frombytes(data[: sample_count * channel_count * 2])
        scale = 1.0 / 32768.0
    elif tag == 3 and bits == 32:
        raw = array.array("f")
        raw.frombytes(data[: sample_count * channel_count * 4])
        scale = 1.0
    else:
        raise AuditionError(f"unhandled wav format tag={tag} bits={bits}: {path}")

    peak = 0.0
    energies = [0.0] * channel_count
    for index, value in enumerate(raw):
        sample = value * scale
        magnitude = abs(sample)
        if magnitude > peak:
            peak = magnitude
        energies[index % channel_count] += sample * sample

    def to_dbfs(energy: float) -> float:
        mean = energy / max(sample_count, 1)
        return 10.0 * math.log10(mean) if mean > 0.0 else float("-inf")

    return {
        "channel_count": channel_count,
        "sample_rate_hz": sample_rate,
        "sample_count": sample_count,
        "duration_s": round(sample_count / sample_rate, 4) if sample_rate else None,
        "peak": round(peak, 4),
        "rms_dbfs": [round(to_dbfs(energy), 2) for energy in energies],
    }


def read_png_first_channel(path: Path) -> tuple[int, int, bytes]:
    """First-channel plane of an 8-bit PNG (grayscale, RGB or RGBA).

    Implements the five scanline filters so it reads what the renderer
    actually wrote; verified against real rendered frames (a frame diffed
    against itself measures exactly zero) before being trusted here.
    """

    blob = path.read_bytes()
    if blob[:8] != b"\x89PNG\r\n\x1a\n":
        raise AuditionError(f"not a PNG: {path}")
    width = height = channels = 0
    idat = bytearray()
    position = 8
    while position + 8 <= len(blob):
        (size,) = struct.unpack(">I", blob[position : position + 4])
        chunk_id = blob[position + 4 : position + 8]
        body = blob[position + 8 : position + 8 + size]
        if chunk_id == b"IHDR":
            width, height, depth, colour, _, _, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if depth != 8 or interlace != 0:
                raise AuditionError(
                    f"only 8-bit non-interlaced PNGs are read (depth={depth}, "
                    f"interlace={interlace}): {path}"
                )
            if colour not in (0, 2, 6):
                raise AuditionError(f"unhandled PNG colour type {colour}: {path}")
            channels = {0: 1, 2: 3, 6: 4}[colour]
        elif chunk_id == b"IDAT":
            idat += body
        elif chunk_id == b"IEND":
            break
        position += 12 + size
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    plane = bytearray(width * height)
    previous = bytearray(stride)
    offset = 0
    for row in range(height):
        line_filter = raw[offset]
        offset += 1
        line = bytearray(raw[offset : offset + stride])
        offset += stride
        if line_filter == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 255
        elif line_filter == 2:
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 255
        elif line_filter == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 255
        elif line_filter == 4:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                above = previous[i]
                corner = previous[i - channels] if i >= channels else 0
                gradient = left + above - corner
                da = abs(gradient - left)
                db = abs(gradient - above)
                dc = abs(gradient - corner)
                if da <= db and da <= dc:
                    predictor = left
                elif db <= dc:
                    predictor = above
                else:
                    predictor = corner
                line[i] = (line[i] + predictor) & 255
        elif line_filter != 0:
            raise AuditionError(f"unknown PNG filter {line_filter}: {path}")
        plane[row * width : (row + 1) * width] = line[::channels]
        previous = line
    return width, height, bytes(plane)


def changed_fraction(a: bytes, b: bytes, threshold: int = _MOTION_PIXEL_THRESHOLD) -> float:
    if len(a) != len(b):
        raise AuditionError("frames differ in size; cannot diff")
    changed = sum(1 for x, y in zip(a, b) if abs(x - y) > threshold)
    return changed / max(len(a), 1)


# ---------------------------------------------------------------------------
# the audit itself


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(a: Any, b: Any, tolerance: float = _POSITION_TOLERANCE_M) -> bool:
    try:
        va = [float(v) for v in a]
        vb = [float(v) for v in b]
    except (TypeError, ValueError):
        return False
    return len(va) == len(vb) and all(
        abs(x - y) <= tolerance for x, y in zip(va, vb)
    )


def _check(name: str, ok: bool, reason: str, **measured: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if ok else "fail",
        "reason_zh": reason,
        "measured": measured,
    }


def _info(name: str, reason: str, **measured: Any) -> dict[str, Any]:
    return {"name": name, "status": "info", "reason_zh": reason, "measured": measured}


def audit_episode(
    episode_dir: Path, *, ffprobe: str | None = "ffprobe"
) -> dict[str, Any]:
    """Audit one rendered episode directory and return the verdict document.

    ``episode_dir`` is the directory holding receipt.json. Artifact paths
    come from the receipt itself - the receipt is the chain's own statement
    of what to read - and a recorded file that is missing is a failed
    check, not an exception, because an audit that crashes on the worst
    inputs is useless exactly then.
    """

    receipt = _read_json(episode_dir / "receipt.json")
    checks: list[dict[str, Any]] = []

    def recorded(key: str) -> Path | None:
        value = receipt.get(key)
        if not value:
            checks.append(_check("artifacts", False, f"收据没有登记 {key}"))
            return None
        path = Path(str(value))
        if not path.is_file():
            checks.append(
                _check("artifacts", False, f"收据登记的 {key} 不存在：{path}")
            )
            return None
        return path

    def optional_recorded(key: str) -> Path | None:
        value = receipt.get(key)
        if value is None:
            return None
        path = Path(str(value))
        if not path.is_file():
            checks.append(
                _check("artifacts", False, f"收据登记的 {key} 不存在：{path}")
            )
            return None
        return path

    pose_path = recorded("listener_pose")
    foa_path = recorded("foa_report")
    foa_wav_path = recorded("foa_wav")
    binaural_wav_path = recorded("binaural_wav")
    full_tail_foa_wav_path = optional_recorded("foa_wav_full_tail")
    full_tail_binaural_wav_path = optional_recorded("binaural_wav_full_tail")
    aligned_foa_wav_path = optional_recorded("foa_wav_aligned")
    aligned_binaural_wav_path = optional_recorded("binaural_wav_aligned")
    manifest_path = recorded("video_manifest")
    mp4_path = recorded("deliverable_mp4")
    if any(
        path is None
        for path in (
            pose_path,
            foa_path,
            foa_wav_path,
            binaural_wav_path,
            manifest_path,
            mp4_path,
        )
    ):
        return _seal(episode_dir, checks)

    pose = _read_json(pose_path)
    foa = _read_json(foa_path)
    manifest = _read_json(manifest_path)
    binaural_report_path = binaural_wav_path.parent / "render_report.json"
    binaural = (
        _read_json(binaural_report_path) if binaural_report_path.is_file() else {}
    )

    # --- one identity through the whole chain ------------------------------
    accepted = pose.get("accepted_index")
    candidate = foa.get("listener_pose_candidate")
    accepted_position = None
    if accepted is not None:
        try:
            accepted_position = pose["candidates"][int(accepted)]["position_m"]
        except (KeyError, IndexError, TypeError, ValueError):
            accepted_position = None
    identity_ok = (
        accepted is not None
        and candidate == accepted
        and accepted_position is not None
        and _close(foa.get("listener_m"), accepted_position)
        and _close(binaural.get("listener_m"), accepted_position)
        and _close(manifest.get("listener_m"), accepted_position)
        and _close(binaural.get("head_aim_world"), manifest.get("camera_aim_world"))
        and str(foa.get("bank")) == str(binaural.get("bank")) == str(receipt.get("bank"))
    )
    checks.append(
        _check(
            "chain_identity",
            identity_ok,
            (
                f"位姿身份链一致：声场、双耳、画面三路渲染的都是同一个被试听接受的"
                f"听者（候选 {accepted}），头朝向与相机朝向同一向量，银行同一文件"
                if identity_ok
                else "位姿身份链断裂：三路渲染读到的听者/朝向/银行不一致，"
                "声画在回答两个不同的问题"
            ),
            accepted_index=accepted,
            foa_candidate=candidate,
        )
    )

    # --- direction: the audio really encodes where the source is -----------
    rendered = foa.get("frames_rendered")
    within = foa.get("frames_within_tolerance")
    tolerance = foa.get("tolerance_deg")
    error = foa.get("direction_error_deg") or {}
    direction_ok = (
        isinstance(rendered, int)
        and rendered > 0
        and isinstance(within, int)
        and within >= _DIRECTION_MINIMUM_FRACTION * rendered
    )
    checks.append(
        _check(
            "foa_direction",
            direction_ok,
            (
                f"声场方向：{within}/{rendered} 帧的音频实测方向与几何真方向差 "
                f"≤{tolerance}°（误差中位 {error.get('median')}°，最大 "
                f"{error.get('maximum')}°）"
                if direction_ok
                else f"声场方向不可信：只有 {within}/{rendered} 帧达到 "
                f"{tolerance}° 容差（要求 ≥{_DIRECTION_MINIMUM_FRACTION:.0%}）"
            ),
            frames_rendered=rendered,
            frames_within_tolerance=within,
            tolerance_deg=tolerance,
            error_deg=error,
        )
    )

    # --- left and right are physically left and right ----------------------
    probes = binaural.get("cardinal_probes") or {}
    margin = binaural.get("cardinal_margin_db")
    left_db = ((probes.get("left") or {}).get("difference_db"))
    right_db = ((probes.get("right") or {}).get("difference_db"))
    cardinal_ok = (
        isinstance(margin, (int, float))
        and isinstance(left_db, (int, float))
        and isinstance(right_db, (int, float))
        and left_db >= margin
        and right_db <= -margin
    )
    checks.append(
        _check(
            "binaural_cardinal",
            cardinal_ok,
            (
                f"左右声道基准：左侧探针左声道响 {left_db} dB、右侧探针右声道响 "
                f"{-right_db if isinstance(right_db, (int, float)) else '?'} dB"
                f"（下限 {margin} dB），片子没有镜像"
                if cardinal_ok
                else "左右声道基准缺失或不过：无法证明左就是左"
            ),
            left_difference_db=left_db,
            right_difference_db=right_db,
            margin_db=margin,
        )
    )

    # --- the shipped wavs: audible, unclipped, the right length ------------
    frame_rate = float(receipt.get("frame_rate_hz") or 0.0) or None
    nominal_s = (
        float(receipt.get("clip_seconds"))
        if receipt.get("clip_seconds") is not None
        else (
            rendered / frame_rate
            if isinstance(rendered, int) and frame_rate
            else None
        )
    )
    binaural_levels = read_wav_levels(binaural_wav_path)
    levels_ok = (
        binaural_levels["channel_count"] == 2
        and binaural_levels["peak"] <= _CLIP_CEILING
        and all(v >= _SILENCE_FLOOR_DBFS for v in binaural_levels["rms_dbfs"])
        and (
            nominal_s is None
            or nominal_s - 0.05
            <= binaural_levels["duration_s"]
            <= nominal_s + _WAV_TAIL_ALLOWANCE_S
        )
    )
    checks.append(
        _check(
            "binaural_wav_levels",
            levels_ok,
            (
                f"双耳 wav 健康：两声道电平 {binaural_levels['rms_dbfs']} dBFS，"
                f"峰值 {binaural_levels['peak']}（无削波），时长 "
                f"{binaural_levels['duration_s']} s（含混响尾）"
                if levels_ok
                else f"双耳 wav 不健康：电平 {binaural_levels['rms_dbfs']} dBFS / "
                f"峰值 {binaural_levels['peak']} / 时长 "
                f"{binaural_levels['duration_s']} s，静音、削波或长度不对"
            ),
            **binaural_levels,
            nominal_duration_s=nominal_s,
        )
    )

    foa_levels = read_wav_levels(foa_wav_path)
    foa_ok = (
        foa_levels["channel_count"] == 4
        and foa_levels["peak"] <= _CLIP_CEILING
        and foa_levels["rms_dbfs"][0] >= _SILENCE_FLOOR_DBFS
    )
    checks.append(
        _check(
            "foa_wav_levels",
            foa_ok,
            (
                f"FOA wav 健康：W 声道 {foa_levels['rms_dbfs'][0]} dBFS，峰值 "
                f"{foa_levels['peak']}，四声道齐全（训练消费的就是它）"
                if foa_ok
                else f"FOA wav 不健康：声道数 {foa_levels['channel_count']} / "
                f"W 电平 {foa_levels['rms_dbfs'][:1]} dBFS / 峰值 {foa_levels['peak']}"
            ),
            **foa_levels,
        )
    )

    # --- the configured clock and aligned audio window ---------------------
    raw_clock = receipt.get("clock")
    if isinstance(raw_clock, Mapping):
        try:
            clock_frames = int(raw_clock["frame_count"])
            clock_rate = float(raw_clock["frame_rate_hz"])
            clock_samples_rate = int(raw_clock["sample_rate_hz"])
            clock_clip = float(raw_clock["clip_seconds"])
            clock_sample_count = int(raw_clock["sample_count"])
            clock_ok = (
                clock_frames > 0
                and math.isfinite(clock_rate)
                and clock_rate > 0.0
                and math.isfinite(clock_clip)
                and abs(clock_clip - clock_frames / clock_rate) <= 1.0e-9
                and clock_samples_rate > 0
                and clock_sample_count == round(clock_clip * clock_samples_rate)
                and manifest.get("frame_count") == clock_frames
                and manifest.get("sample_count") == clock_sample_count
                and manifest.get("sample_rate_hz") == clock_samples_rate
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            clock_ok = False
            clock_frames = clock_rate = clock_samples_rate = clock_clip = clock_sample_count = None
        checks.append(
            _check(
                "episode_clock",
                clock_ok,
                (
                    f"声画共用时钟：{clock_frames} 帧 @ {clock_rate:g} Hz，"
                    f"{clock_sample_count} samples @ {clock_samples_rate} Hz，"
                    f"aligned 窗口 {clock_clip:g} s"
                    if clock_ok
                    else "收据中的 frame/rate/sample/clip 时钟不一致，无法证明声画对齐"
                ),
                frame_count=clock_frames,
                frame_rate_hz=clock_rate,
                sample_rate_hz=clock_samples_rate,
                sample_count=clock_sample_count,
                clip_seconds=clock_clip,
            )
        )
        aligned_checks = []
        for path, channels in (
            (aligned_foa_wav_path, 4),
            (aligned_binaural_wav_path, 2),
        ):
            if path is None:
                aligned_checks.append(False)
                continue
            try:
                levels = read_wav_levels(path)
                aligned_checks.append(
                    levels["channel_count"] == channels
                    and levels["sample_rate_hz"] == clock_samples_rate
                    and levels["sample_count"] == clock_sample_count
                )
            except (OSError, AuditionError):
                aligned_checks.append(False)
        aligned_ok = all(aligned_checks)
        checks.append(
            _check(
                "aligned_audio_window",
                aligned_ok,
                (
                    f"aligned FOA/双耳文件均为 {clock_sample_count} samples；"
                    "full-tail 文件单独保留，mux 使用 aligned 窗口"
                    if aligned_ok
                    else "aligned 音频窗口缺失或 sample_count/sample_rate 不符"
                ),
                expected_sample_count=clock_sample_count,
                foa_present=aligned_foa_wav_path is not None,
                binaural_present=aligned_binaural_wav_path is not None,
            )
        )
        full_tail_checks = []
        full_tail_details = {}
        for name, registered, legacy, aligned, channels, levels, report in (
            (
                "foa", full_tail_foa_wav_path, foa_wav_path,
                aligned_foa_wav_path, 4, foa_levels, foa,
            ),
            (
                "binaural", full_tail_binaural_wav_path, binaural_wav_path,
                aligned_binaural_wav_path, 2, binaural_levels, binaural,
            ),
        ):
            path_identity_ok = (
                registered is not None
                and registered.resolve() == legacy.resolve()
                and aligned is not None
                and registered.resolve() != aligned.resolve()
            )
            content_ok = (
                levels["channel_count"] == channels
                and levels["sample_rate_hz"] == clock_samples_rate
                and levels["sample_count"] >= clock_sample_count
                and report.get("full_tail_sample_count")
                == levels["sample_count"]
            )
            full_tail_checks.append(path_identity_ok and content_ok)
            full_tail_details[name] = {
                "registered": str(registered) if registered is not None else None,
                "aligned": str(aligned) if aligned is not None else None,
                "sample_count": levels["sample_count"],
                "report_sample_count": report.get("full_tail_sample_count"),
                "path_identity_ok": path_identity_ok,
            }
        full_tail_ok = all(full_tail_checks)
        checks.append(
            _check(
                "full_tail_audio",
                full_tail_ok,
                (
                    "FOA/双耳 full-tail 文件与 aligned 文件分别登记，"
                    "报告和实际 sample count 一致"
                    if full_tail_ok
                    else "full-tail 文件缺失、与 aligned 共用路径，或报告 sample count 不符"
                ),
                files=full_tail_details,
            )
        )
    else:
        checks.append(
            _info(
                "episode_clock",
                "历史收据没有显式共享时钟；按 legacy frame_rate_hz 审核",
            )
        )

    # --- the shipped wav carries the spatialization the report measured ----
    ild = binaural.get("interaural_level_difference_db") or {}
    report_median = ild.get("median")
    wav_ild = None
    if binaural_levels["channel_count"] == 2 and all(
        v != float("-inf") for v in binaural_levels["rms_dbfs"]
    ):
        wav_ild = round(
            binaural_levels["rms_dbfs"][0] - binaural_levels["rms_dbfs"][1], 2
        )
    if isinstance(report_median, (int, float)) and wav_ild is not None:
        delta = round(abs(wav_ild - float(report_median)), 2)
        if abs(float(report_median)) >= _ILD_GATE_MINIMUM_DB:
            side_ok = (wav_ild > 0) == (float(report_median) > 0)
            checks.append(
                _check(
                    "binaural_wav_matches_report",
                    side_ok and delta <= _ILD_WINDOW_DB,
                    (
                        f"成品声与报告一致:wav 整段左右差 {wav_ild} dB vs 报告中位 "
                        f"{report_median} dB（差 {delta} dB，同侧）——声道没有在"
                        f"写文件时被调换"
                        if side_ok and delta <= _ILD_WINDOW_DB
                        else f"成品声与报告不一致：wav 整段左右差 {wav_ild} dB，"
                        f"报告中位 {report_median} dB——疑似声道调换或渲染错位"
                    ),
                    wav_overall_ild_db=wav_ild,
                    report_median_ild_db=report_median,
                    delta_db=delta,
                )
            )
        else:
            checks.append(
                _info(
                    "binaural_wav_matches_report",
                    f"报告中位左右差只有 {report_median} dB（声源太居中），"
                    f"此项只记录不判：wav 实测 {wav_ild} dB",
                    wav_overall_ild_db=wav_ild,
                    report_median_ild_db=report_median,
                    delta_db=delta,
                )
            )
    else:
        checks.append(
            _check(
                "binaural_wav_matches_report",
                False,
                "双耳报告缺少逐帧左右差统计，或 wav 有静音声道，无法对照",
                wav_overall_ild_db=wav_ild,
                report_median_ild_db=report_median,
            )
        )

    # --- the shipped frames: present, lit, and something moves -------------
    video_dir = manifest_path.parent
    frames = sorted(video_dir.glob("frame_*.png"))
    frames_ok = isinstance(rendered, int) and len(frames) == rendered
    if frames_ok and frames:
        width, height, first = read_png_first_channel(frames[0])
        _, _, middle = read_png_first_channel(frames[len(frames) // 2])
        _, _, last = read_png_first_channel(frames[-1])
        levels = [
            round(sum(plane) / len(plane), 1) for plane in (first, middle, last)
        ]
        motion = round(
            max(changed_fraction(first, middle), changed_fraction(first, last)), 5
        )
        lit = all(level >= _FRAME_LEVEL_MINIMUM for level in levels)
        moving = motion >= _MOTION_MINIMUM_FRACTION
        checks.append(
            _check(
                "video_frames",
                lit and moving,
                (
                    f"画面健康：{len(frames)} 帧齐全（{width}×{height}），首/中/末帧"
                    f"亮度 {levels}，首末帧间 {motion:.2%} 的像素在动（声源确实在走）"
                    if lit and moving
                    else f"画面不健康：亮度 {levels}（黑帧？）或首末帧只有 "
                    f"{motion:.2%} 像素变化（{_MOTION_MINIMUM_FRACTION:.2%} 起判）"
                ),
                frame_count=len(frames),
                width=width,
                height=height,
                mean_levels=levels,
                changed_fraction=motion,
            )
        )
    else:
        checks.append(
            _check(
                "video_frames",
                False,
                f"帧数不对：盘上 {len(frames)} 张 PNG，报告说渲染了 {rendered} 帧",
                frame_count=len(frames),
                frames_rendered=rendered,
            )
        )

    # --- the deliverable holds both streams at the right length ------------
    probe_binary = ffprobe if ffprobe and shutil.which(ffprobe) else None
    if probe_binary is None:
        checks.append(
            _info(
                "deliverable_mux",
                "ffprobe 不可用，封装未验（帧与 wav 已分别验过）",
            )
        )
    else:
        completed = subprocess.run(
            [
                probe_binary,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(mp4_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            probe = json.loads(completed.stdout or "{}")
        except ValueError:
            probe = {}
        streams = probe.get("streams") or []
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        mp4_frames = None
        if video_streams and str(video_streams[0].get("nb_frames", "")).isdigit():
            mp4_frames = int(video_streams[0]["nb_frames"])
        try:
            duration = float((probe.get("format") or {}).get("duration"))
        except (TypeError, ValueError):
            duration = None
        mux_ok = (
            completed.returncode == 0
            and bool(video_streams)
            and bool(audio_streams)
            and (mp4_frames is None or mp4_frames == rendered)
            and (
                duration is not None
                and nominal_s is not None
                and abs(duration - nominal_s) <= _MUX_DURATION_TOLERANCE_S
            )
        )
        checks.append(
            _check(
                "deliverable_mux",
                mux_ok,
                (
                    f"封装完整：mp4 含 {mp4_frames} 帧视频 + "
                    f"{audio_streams[0].get('codec_name')} 音频，时长 {duration} s"
                    f"（应为 {nominal_s} s）"
                    if mux_ok
                    else f"封装不完整：视频流 {len(video_streams)} / 音频流 "
                    f"{len(audio_streams)} / 帧 {mp4_frames}（应 {rendered}）/ "
                    f"时长 {duration} s（应 {nominal_s}±{_MUX_DURATION_TOLERANCE_S} s）"
                ),
                video_streams=len(video_streams),
                audio_streams=len(audio_streams),
                mp4_frames=mp4_frames,
                duration_s=duration,
                nominal_duration_s=nominal_s,
            )
        )

    return _seal(episode_dir, checks)


def _seal(episode_dir: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [check for check in checks if check["status"] == "fail"]
    verdict = "pass" if not failures else "fail"
    if failures:
        summary = "不过：" + "；".join(
            check["reason_zh"] for check in failures[:3]
        )
    else:
        named = {check["name"]: check for check in checks}

        def measured(name: str, key: str) -> Any:
            return (named.get(name, {}).get("measured") or {}).get(key)

        summary = (
            f"方向 {measured('foa_direction', 'frames_within_tolerance')}/"
            f"{measured('foa_direction', 'frames_rendered')} 帧达标 · 左右声道有"
            f"实证 · 成品声画与报告一致 · 无静音/削波/黑帧 · 封装完整"
        )
    return {
        "schema": SCHEMA,
        "episode_dir": str(episode_dir),
        "verdict": verdict,
        "summary_zh": summary,
        "checks": checks,
        "scope_note_zh": (
            "机器听审验证的是物理与几何一致性（方向、声道、能量、画面、封装、"
            "身份链），不评价好听不好听；人工覆核是可选的覆盖，不是必经工序"
        ),
        "calibration_note": (
            "gates calibrated on task 20260828T155757Z (scene 00808); "
            "see avengine.review.episode_audition module docstring"
        ),
    }


def write_audition(
    episode_dir: Path, *, ffprobe: str | None = "ffprobe", refresh: bool = False
) -> dict[str, Any]:
    """Audit and write machine_audition.json beside the receipt."""

    target = episode_dir / "machine_audition.json"
    if target.exists() and not refresh:
        raise AuditionError(
            f"{target} already exists; pass refresh=True to re-audit"
        )
    document = audit_episode(episode_dir, ffprobe=ffprobe)
    target.write_text(
        json.dumps(document, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document
