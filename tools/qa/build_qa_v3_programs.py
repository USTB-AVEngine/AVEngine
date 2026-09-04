#!/usr/bin/env python3
"""Per-point audio-program generator for the v3 pilot (work order item 1.2).

拆掉"发声恒在第 4 帧"的病根:qa_v2 用 8 个预制固定 program(全批同一
套 onset),本工具给**每个采样点生成专属 program**——发声时刻在可行域
内均匀随机、每集至少 3 声、两只都至少叫一声、锚事件(全片最后一声)
之后留静默尾窗(错时题的提问窗,窗内两只都无事件)。

事件规划约束(全部显式参数,写进批报告):
  FIRST_MIN     首声最早时刻(默认 0.3s)——防事件贴片头;
  GAP_MIN       相邻事件间隔下限(默认 0.3s);
  TAIL_SILENCE  锚事件结束到片尾的静默下限(默认 1.5s);
  n_events      每点 3 或 4(种子定),归属交替(first_slot 起手),
                保证两只各 ≥1 声(⑧的"单叫退化"防线);
  历史路径事件时长固定 0.3s,素材窗 [0.2s, 0.5s)。新路径每个事件带
  duration_samples,素材窗为整段 clip(一声一个文件)。

产物:每点一个 `<program_id>.json`(结构与 schema 同在产 program:
timeline 常量块、sequential_sources 模式、两端点、封印
program_content_sha256 用引擎自己的 canonical_json_sha256 计算)+
伴生 `<program_id>.plan.json`(锚事件元数据:锚槽位、锚起止、静默尾
长——供错时采样过滤器与出题器消费,不塞进 program 本体以免破坏
schema)。每个产物过 jsonschema 校验,失败即停;输出目录 no-clobber。

批级自检:首声 onset 的帧分布必须覆盖 ≥3 个不同的 15 帧桶(旧病根的
回归锚:全批 onset 挤同一帧即报错)。research_candidate。

请求清单格式(JSON 列表):
  {"point_id": "v3p001", "pair_kind": "dog"|"human",
   "endpoint_1": "...", "endpoint_2": "...",
   "sound_asset_id": "...", "first_slot": "source1"|"source2"}
用法:
  build_qa_v3_programs.py --requests R.json --seed S --out-dir DIR
      [--first-min-s 0.3] [--gap-min-s 0.3] [--tail-silence-s 1.5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping

REPO = Path(__file__).resolve().parents[2]
AUDIO_PROGRAM_SCHEMA_PATH = REPO / "schemas/m6_audio_program_v1.schema.json"
sys.path.insert(0, str(REPO / "src"))

from avengine.contracts.json_io import canonical_json_sha256  # noqa: E402

TIMELINE_KEYS = (
    "time_base_hz", "ticks_per_frame", "video_fps", "frame_count",
    "sample_rate_hz", "ticks_per_sample", "sample_count",
)
PROGRAM_REQUEST_KEYS = (
    "linear_gain", "fade_samples", "mode", "timeline",
    "normalization_policy", "render_source_stem",
    "source_specific_stems", "admission_state",
)


def _require_request(request: dict, key: str):
    if key not in request:
        raise ValueError(f"request missing {key}")
    return request[key]


def _require_param(params: Mapping, key: str):
    if key not in params:
        raise ValueError(f"params missing {key}")
    return params[key]


def load_audio_program_schema() -> dict:
    return json.loads(AUDIO_PROGRAM_SCHEMA_PATH.read_text(encoding="utf-8"))


def linear_gain_schema_bounds() -> dict:
    spec = load_audio_program_schema()["$defs"]["event"]["properties"]["linear_gain"]
    missing = [key for key in ("exclusiveMinimum", "maximum") if key not in spec]
    if missing:
        raise ValueError(f"schema linear_gain missing {missing}")
    return {
        "exclusiveMinimum": float(spec["exclusiveMinimum"]),
        "maximum": float(spec["maximum"]),
    }


def _schema_number_text(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return repr(number)


def check_program_linear_gain(value: float) -> float:
    """Refuse a gain the frozen program schema would not admit.

    The schema maximum is inclusive and means: prepared clips are already
    peak-normalized, so the program may only attenuate.
    """
    bounds = linear_gain_schema_bounds()
    lo = bounds["exclusiveMinimum"]
    hi = bounds["maximum"]
    if not (lo < float(value) <= hi):
        raise ValueError(
            f"PROGRAM_LINEAR_GAIN={value} is outside schema linear_gain "
            f"bounds exclusiveMinimum={_schema_number_text(lo)} "
            f"maximum={_schema_number_text(hi)}"
        )
    return float(value)


def validate_m6_audio_program(doc, *, schema_path: Path | None = None) -> None:
    import jsonschema
    schema_file = Path(schema_path) if schema_path is not None else AUDIO_PROGRAM_SCHEMA_PATH
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    err = errors[0]
    path = list(err.absolute_path)
    message = err.message
    if path and path[-1] == "linear_gain":
        bounds = linear_gain_schema_bounds()
        message = (
            f"linear_gain={err.instance} is outside schema linear_gain bounds "
            f"exclusiveMinimum={_schema_number_text(bounds['exclusiveMinimum'])} "
            f"maximum={_schema_number_text(bounds['maximum'])}: {err.message}"
        )
    raise ValueError(f"audio program schema violation: {message}")


def require_dry_canvas_source_mode(params: Mapping, *, owner: str) -> None:
    """These assemblers still emit a shared dry canvas. event_pool is refused."""
    if "SOUND_SOURCE_MODE" not in params:
        raise ValueError(f"{owner}: params missing SOUND_SOURCE_MODE")
    mode = str(params["SOUND_SOURCE_MODE"])
    if mode != "dry_canvas_window":
        raise ValueError(
            f"{owner} currently supports only SOUND_SOURCE_MODE="
            f"dry_canvas_window, got {mode!r}")


def program_request_fields(params: Mapping, *, include_mode: bool = True) -> dict:
    """Copy program policy from a params file. None of these have code defaults.

    Pass include_mode=False when the caller derives AudioProgram.mode from
    the event list instead of PROGRAM_MODE.
    """
    rate = int(_require_param(params, "SAMPLE_RATE_HZ"))
    clip_seconds = float(_require_param(params, "CLIP_SECONDS"))
    if rate <= 0 or clip_seconds <= 0:
        raise ValueError("SAMPLE_RATE_HZ and CLIP_SECONDS must be positive")
    fields = {
        "linear_gain": check_program_linear_gain(
            float(_require_param(params, "PROGRAM_LINEAR_GAIN"))),
        "fade_samples": int(_require_param(params, "PROGRAM_FADE_SAMPLES")),
        "timeline": {
            "time_base_hz": int(_require_param(params, "TIME_BASE_HZ")),
            "ticks_per_frame": int(_require_param(params, "TICKS_PER_FRAME")),
            "video_fps": int(_require_param(params, "VIDEO_FPS")),
            "frame_count": int(_require_param(params, "FRAME_COUNT")),
            "sample_rate_hz": rate,
            "ticks_per_sample": int(_require_param(params, "TICKS_PER_SAMPLE")),
            "sample_count": int(round(clip_seconds * rate)),
        },
        "normalization_policy": str(
            _require_param(params, "PROGRAM_NORMALIZATION_POLICY")),
        "render_source_stem": bool(
            _require_param(params, "PROGRAM_RENDER_SOURCE_STEM")),
        "source_specific_stems": bool(
            _require_param(params, "PROGRAM_SOURCE_SPECIFIC_STEMS")),
        "admission_state": str(
            _require_param(params, "PROGRAM_ADMISSION_STATE")),
    }
    if include_mode:
        fields["mode"] = str(_require_param(params, "PROGRAM_MODE"))
    return fields


def dry_canvas_window_fields(params: Mapping) -> dict:
    rate = int(_require_param(params, "SAMPLE_RATE_HZ"))
    return {
        "event_duration_samples": int(round(
            float(_require_param(params, "EVENT_SECONDS")) * rate)),
        "source_start_sample": int(
            _require_param(params, "DRY_CANVAS_SOURCE_START_SAMPLE")),
        "source_end_sample_exclusive": int(
            _require_param(params, "DRY_CANVAS_SOURCE_END_SAMPLE_EXCLUSIVE")),
    }


def dry_canvas_fields(params: Mapping) -> dict:
    return {
        "sound_asset_id": str(_require_param(params, "SOUND_ASSET")),
        **dry_canvas_window_fields(params),
    }


def plan_timebase(params: Mapping) -> dict:
    rate = int(_require_param(params, "SAMPLE_RATE_HZ"))
    clip_seconds = float(_require_param(params, "CLIP_SECONDS"))
    event_seconds = float(_require_param(params, "EVENT_SECONDS"))
    if rate <= 0 or clip_seconds <= 0 or event_seconds <= 0:
        raise ValueError("SAMPLE_RATE_HZ, CLIP_SECONDS, EVENT_SECONDS must be positive")
    return {
        "sample_rate_hz": rate,
        "sample_count": int(round(clip_seconds * rate)),
        "event_len_samples": int(round(event_seconds * rate)),
    }


def _timeline(request: dict) -> dict:
    timeline = _require_request(request, "timeline")
    if not isinstance(timeline, dict):
        raise ValueError("request timeline must be an object")
    missing = [key for key in TIMELINE_KEYS if key not in timeline]
    if missing:
        raise ValueError(f"request timeline missing {missing}")
    return {
        "time_base_hz": int(timeline["time_base_hz"]),
        "ticks_per_frame": int(timeline["ticks_per_frame"]),
        "video_fps": int(timeline["video_fps"]),
        "frame_count": int(timeline["frame_count"]),
        "sample_rate_hz": int(timeline["sample_rate_hz"]),
        "ticks_per_sample": int(timeline["ticks_per_sample"]),
        "sample_count": int(timeline["sample_count"]),
    }


def _canvas_window(request: dict) -> tuple[int, int, int]:
    missing = [key for key in (
        "event_duration_samples",
        "source_start_sample",
        "source_end_sample_exclusive",
    ) if key not in request]
    if missing:
        raise ValueError(f"request missing {missing}")
    duration = int(request["event_duration_samples"])
    start = int(request["source_start_sample"])
    end = int(request["source_end_sample_exclusive"])
    if duration <= 0 or end <= start:
        raise ValueError("request source window is empty")
    return duration, start, end


def _rng(seed: str, *parts: str):
    import numpy as np
    digest = hashlib.sha256(("\0".join((seed,) + parts)).encode()).hexdigest()
    return np.random.default_rng(int(digest[:12], 16))


def plan_events(seed: str, point_id: str, first_slot: str, *,
                first_min_s: float, gap_min_s: float, tail_silence_s: float,
                event_len_samples: int,
                sample_rate_hz: int, sample_count: int,
                first_call_bands: list[float] | None = None,
                min_first_call_gap_s: float | None = None,
                target_first_bands: tuple[int, int] | None = None):
    """返回 (事件列表[(slot, start_sample)], 锚元数据)。

    可选的 card8 约束(设计冒烟的教训:这两条必须在 program 规划层满足,
    换路线的重试对它们无能为力):两个槽位的**首叫**须落在
    first_call_bands 的不同带,且相隔超过 min_first_call_gap_s。

    `target_first_bands=(b1, b2)` 是**带优先调度**(codex 审阅裁定):先
    给这一点分配答案带,再在带内采样,而不是先采样再看落在哪带——后者
    让答案带的分布由可行域形状决定(run01 实测 54/60/73/53,按外观拆开
    后黑白狗 22/21/48/29,只看外观猜第三带即得 40%)。因为事件按槽位
    交替、时间递增,首叫必然 t1 < t2,故只有 b1 < b2 的有序带对可达;
    调用方按有序带对均匀配额分配,两个槽位的答案带边际分布即自动配平。
    """
    rng = _rng(seed, point_id, "events")
    other = "source2" if first_slot == "source1" else "source1"
    lo = int(first_min_s * sample_rate_hz)
    gap = int(gap_min_s * sample_rate_hz)
    hi = sample_count - int(tail_silence_s * sample_rate_hz) - event_len_samples
    step = event_len_samples + gap
    if target_first_bands is not None:
        if first_call_bands is None:
            raise ValueError("target_first_bands requires first_call_bands")
        b1, b2 = target_first_bands
        if not 0 <= b1 < b2 <= len(first_call_bands) - 2:
            raise ValueError(f"{point_id}: unreachable band pair {b1},{b2} "
                             "(events alternate in time, so b1 < b2)")
        starts, n_events = _plan_banded_starts(
            rng, point_id, first_call_bands, (b1, b2), lo, hi, step,
            min_first_call_gap_s, sample_rate_hz)
        slots = [first_slot if i % 2 == 0 else other
                 for i in range(n_events)]
        return _finish_plan(slots, starts, n_events, event_len_samples,
                            sample_count)
    n_events = int(rng.integers(3, 5))  # 3 或 4
    slots = [first_slot if i % 2 == 0 else other for i in range(n_events)]
    if hi - lo < (n_events - 1) * step:
        raise ValueError(f"{point_id}: infeasible constraints "
                         f"(window {hi - lo} < needed {(n_events - 1) * step})")

    def _band(t_s: float):
        b = first_call_bands
        if t_s < b[0] or t_s > b[-1]:
            return None
        for i in range(len(b) - 1):
            if b[i] <= t_s < b[i + 1]:
                return i
        return len(b) - 2

    # 顺序条件采样:每一步在当前可行区间内均匀取。教训(自检抓到的
    # 设计缺陷):"采 n 个均匀样本再排序"会让首声=n 个样本的最小值,
    # 分布挤向低端,批内首声全落前几十帧——"首声可预测"换形式复活。
    # 条件采样让首声的边际分布在 [lo, hi-(n-1)step] 上真均匀。
    starts: list[int] = []
    for _attempt in range(300):
        starts = []
        cursor = lo
        for i in range(n_events):
            remaining = n_events - 1 - i
            hi_i = hi - remaining * step
            starts.append(int(rng.integers(cursor, hi_i + 1)))
            cursor = starts[-1] + step
        if first_call_bands is None and min_first_call_gap_s is None:
            break
        # 首叫 = 交替序列的前两个事件(各槽位第一次)
        t1, t2 = starts[0] / sample_rate_hz, starts[1] / sample_rate_hz
        gap_ok = (min_first_call_gap_s is None
                  or abs(t2 - t1) > min_first_call_gap_s)
        band_ok = (first_call_bands is None or _band(t1) != _band(t2))
        if gap_ok and band_ok:
            break
    else:
        raise RuntimeError(f"{point_id}: no onset layout satisfying first-call "
                           f"band/gap constraints in 300 attempts")
    return _finish_plan(slots, starts, n_events, event_len_samples,
                        sample_count)


def _plan_banded_starts(rng, point_id, bands, target, lo, hi, step,
                        min_first_call_gap_s, sample_rate_hz):
    """带优先:两个首叫先各自锁进目标带,其余事件再条件采样。

    带内仍是条件均匀采样(保住"首叫时刻在带内不可预测");事件数 3/4
    的选择随目标带对的可行性走——晚带对需要更短的事件序列才塞得下。
    """
    b1, b2 = target
    lo1, hi1 = bands[b1] * sample_rate_hz, bands[b1 + 1] * sample_rate_hz
    lo2, hi2 = bands[b2] * sample_rate_hz, bands[b2 + 1] * sample_rate_hz
    gap_samples = (0 if min_first_call_gap_s is None
                   else int(min_first_call_gap_s * sample_rate_hz))
    for n_events in [int(v) for v in rng.permutation([3, 4])]:
        t1_lo = max(lo, int(lo1))
        t1_hi = min(hi - (n_events - 1) * step, int(hi1) - 1)
        if t1_lo > t1_hi:
            continue
        for _attempt in range(200):
            t1 = int(rng.integers(t1_lo, t1_hi + 1))
            t2_lo = max(t1 + step, int(lo2), t1 + gap_samples + 1)
            t2_hi = min(hi - (n_events - 2) * step, int(hi2) - 1)
            if t2_lo > t2_hi:
                continue
            starts = [t1, int(rng.integers(t2_lo, t2_hi + 1))]
            cursor = starts[-1] + step
            ok = True
            for i in range(2, n_events):
                hi_i = hi - (n_events - 1 - i) * step
                if cursor > hi_i:
                    ok = False
                    break
                starts.append(int(rng.integers(cursor, hi_i + 1)))
                cursor = starts[-1] + step
            if ok:
                return starts, n_events
    raise RuntimeError(f"{point_id}: no onset layout lands the first calls in "
                       f"bands {target} (tried 3 and 4 events)")


def _finish_plan(slots, starts, n_events, event_len_samples, sample_count):
    events = list(zip(slots, starts))
    anchor_slot, anchor_start = events[-1]
    anchor = {"anchor_event_index": n_events - 1,
              "anchor_slot": anchor_slot,
              "anchor_start_sample": anchor_start,
              "anchor_end_sample": anchor_start + event_len_samples,
              "tail_silence_samples": sample_count - (
                  anchor_start + event_len_samples),
              "n_events": n_events,
              "per_slot_counts": {s: slots.count(s) for s in set(slots)}}
    return events, anchor


def build_program(request: dict, events, *, revision: str) -> dict:
    missing = [key for key in PROGRAM_REQUEST_KEYS if key not in request]
    if missing:
        raise ValueError(f"request missing {missing}")
    timeline = _timeline(request)
    ticks_per_sample = int(timeline["ticks_per_sample"])
    if "slot_endpoints" in request:
        slot_to_ep = {
            str(slot): str(endpoint)
            for slot, endpoint in request["slot_endpoints"].items()}
        if len(slot_to_ep) < 2 or len(set(slot_to_ep.values())) != len(slot_to_ep):
            raise ValueError("slot_endpoints must bind at least two unique endpoints")
        endpoints = list(slot_to_ep.values())
    else:
        endpoints = [request["endpoint_1"], request["endpoint_2"]]
        slot_to_ep = {"source1": endpoints[0], "source2": endpoints[1]}
    ev_rows = []
    seen_event_ids: set[str] = set()
    for i, event in enumerate(events):
        explicit_event_id: str | None = None
        if isinstance(event, dict):
            for key in ("slot", "start_sample", "duration_samples",
                        "sound_asset_id", "source_start_sample",
                        "source_end_sample_exclusive"):
                if key not in event:
                    raise ValueError(f"program event missing {key}")
            if "event_id" in event:
                raw_event_id = event["event_id"]
                if (not isinstance(raw_event_id, str)
                        or not raw_event_id.strip()):
                    raise ValueError(
                        "program event event_id must be a non-empty string"
                    )
                explicit_event_id = raw_event_id
            slot = str(event["slot"])
            start = int(event["start_sample"])
            duration = int(event["duration_samples"])
            sound_asset_id = event["sound_asset_id"]
            source_start = int(event["source_start_sample"])
            source_end = int(event["source_end_sample_exclusive"])
        elif len(event) == 2:
            slot, start = event
            sound_asset_id = request["sound_asset_id"]
            duration, source_start, source_end = _canvas_window(request)
        elif len(event) == 3:
            slot, start, sound_asset_id = event
            duration, source_start, source_end = _canvas_window(request)
        else:
            raise ValueError(
                "events must be (slot,start), (slot,start,sound_asset_id), "
                "or a dict with duration and source window")
        event_id = explicit_event_id or f"{slot}_event_{i}"
        if event_id in seen_event_ids:
            raise ValueError(f"program event_id must be unique: {event_id!r}")
        seen_event_ids.add(event_id)
        end = start + duration
        ev_rows.append({
            "event_id": event_id,
            "source_endpoint_id": slot_to_ep[slot],
            "sound_asset_id": str(sound_asset_id),
            "start_tick": start * ticks_per_sample,
            "end_tick_exclusive": end * ticks_per_sample,
            "start_sample": start,
            "end_sample_exclusive": end,
            "source_start_sample": source_start,
            "source_end_sample_exclusive": source_end,
            "linear_gain": float(request["linear_gain"]),
            "fade_samples": int(request["fade_samples"]),
            "normalization_policy": str(request["normalization_policy"]),
            "render_source_stem": bool(request["render_source_stem"]),
        })
    doc = {
        "schema": "avengine_m6_audio_program_v1",
        "program_id": f"qa_v3_{request['pair_kind']}_{request['point_id']}_rand_{revision}",
        "revision": revision,
        "mode": str(request["mode"]),
        "timeline": timeline,
        "candidate_source_endpoint_ids": endpoints,
        "events": ev_rows,
        "source_specific_stems": bool(request["source_specific_stems"]),
        "admission_state": str(request["admission_state"]),
    }
    doc["program_content_sha256"] = canonical_json_sha256(doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--first-min-s", type=float, required=True)
    parser.add_argument("--gap-min-s", type=float, required=True)
    parser.add_argument("--tail-silence-s", type=float, required=True)
    parser.add_argument("--event-seconds", type=float, required=True)
    parser.add_argument("--sample-rate-hz", type=int, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--max-top-first-frame-share", type=float, required=True)
    parser.add_argument("--min-onset-buckets", type=int, required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)

    if os.path.exists(args.out_dir):
        print(f"refusing to overwrite existing output dir: {args.out_dir}", file=sys.stderr)
        return 2
    requests = json.load(open(args.requests))

    os.makedirs(args.out_dir)
    onset_frames = []
    rows = []
    for req in requests:
        events, anchor = plan_events(
            args.seed, req["point_id"], req["first_slot"],
            first_min_s=args.first_min_s,
            gap_min_s=args.gap_min_s,
            tail_silence_s=args.tail_silence_s,
            event_len_samples=int(round(args.event_seconds * args.sample_rate_hz)),
            sample_rate_hz=args.sample_rate_hz,
            sample_count=args.sample_count)
        doc = build_program(req, events, revision=args.revision)
        try:
            validate_m6_audio_program(doc, schema_path=args.schema)
        except ValueError as exc:
            print(f"FAIL {req['point_id']}: {exc}", file=sys.stderr)
            return 1
        with open(os.path.join(args.out_dir, doc["program_id"] + ".json"), "w") as fp:
            json.dump(doc, fp, ensure_ascii=False, indent=1)
        plan = {"schema": "avengine_qa_v3_program_plan_v1",
                "status": "research_candidate", "qualification_claim": False,
                "point_id": req["point_id"], "program_id": doc["program_id"],
                "first_slot": req["first_slot"],
                "parameters": {"first_min_s": args.first_min_s,
                               "gap_min_s": args.gap_min_s,
                               "tail_silence_s": args.tail_silence_s},
                **anchor}
        with open(os.path.join(args.out_dir, doc["program_id"] + ".plan.json"), "w") as fp:
            json.dump(plan, fp, ensure_ascii=False, indent=1)
        onset_frames.append(int(
            events[0][1] / args.sample_count * args.frame_count))
        rows.append({"point_id": req["point_id"], "program_id": doc["program_id"],
                     "n_events": anchor["n_events"],
                     "first_onset_frame": onset_frames[-1],
                     "anchor_start_frame": int(anchor["anchor_start_sample"]
                                               / args.sample_count
                                               * args.frame_count),
                     "anchor_slot": anchor["anchor_slot"]})

    # 批级自检:旧病根是"全批首声恒在同一帧",所以最贴切的锚是
    # 单一帧值的占比上限;桶数下限取 2——首声可行域被"≥3 声+间隔+尾窗"
    # 数学地压在片长前 40%(约帧 4–30),散布其中已不可预测,
    # 更宽的桶数要求会误伤合法约束。
    from collections import Counter
    fps = args.frame_count / (args.sample_count / args.sample_rate_hz)
    bucket_width = int(round(fps))
    first_buckets = {f // bucket_width for f in onset_frames}
    anchor_buckets = {r["anchor_start_frame"] // bucket_width for r in rows}
    top_frame_share = max(Counter(onset_frames).values()) / len(rows)
    manifest = {
        "schema": "avengine_qa_v3_program_batch_v1",
        "status": "research_candidate", "qualification_claim": False,
        "seed": args.seed,
        "count": len(rows),
        "first_onset_frame_buckets": sorted(first_buckets),
        "anchor_frame_buckets": sorted(anchor_buckets),
        "top_first_frame_share": round(top_frame_share, 4),
        "rows": rows,
    }
    with open(os.path.join(args.out_dir, "programs_manifest.json"), "w") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=1)
    if (len(requests) >= 12
            and (len(first_buckets) < args.min_onset_buckets
                 or len(anchor_buckets) < args.min_onset_buckets
                 or top_frame_share > args.max_top_first_frame_share)):
        print(f"FAIL: onsets collapse (first={sorted(first_buckets)}, "
              f"anchor={sorted(anchor_buckets)}, top_frame_share="
              f"{top_frame_share:.2f}) — randomization is broken", file=sys.stderr)
        return 1
    print(f"programs={len(rows)} first_buckets={sorted(first_buckets)} "
          f"anchor_buckets={sorted(anchor_buckets)} "
          f"top_first_frame_share={top_frame_share:.2f} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
