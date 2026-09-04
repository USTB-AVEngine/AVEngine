#!/usr/bin/env python3
"""Batch scan: does every capture's camera yaw match its audio listener?

新发现的缺陷是渲染链只交叉校验相机**位置**、不校验**朝向**。既有批次
用的是同一个固定 yaw,理论上不受影响 —— 但"理论上"不是核验。本工具对
一批视觉捕获逐份读出相机位姿,与音频用的 M1 请求听者朝向逐份比对,
报检查数量、最大 yaw 误差与不一致清单。

同时报告**代码版本边界**:音频渲染批跨过了加入 yaw 断言的那次提交,
故按 receipt 落盘时间与提交时间的先后,标出每份音频输出由哪一版代码
产生、新断言从哪一份开始生效。

复用 assert_listener_matches_capture_yaw 的同一套换算(不重复实现)。
research_candidate;输出 no-clobber。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from avengine.dataset.apartment_dynamic_audio import (  # noqa: E402
    captured_static_camera_world_m,
    listener_ue_yaw_deg,
)
from avengine.timeline.current_mp3d_dynamic_audio import (  # noqa: E402
    CurrentMP3DDynamicAudioError,
    listener_pose_from_m1_request,
)


def circular_gap_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures-root", required=True, type=Path)
    parser.add_argument("--m1-request", required=True, type=Path)
    parser.add_argument("--audio-root", type=Path,
                        help="可选:音频输出根,用于标代码版本边界")
    parser.add_argument("--assertion-epoch", type=float, default=None,
                        help="yaw 断言生效的 unix 时间戳(提交时间)")
    parser.add_argument("--tolerance-deg", type=float, default=1.0e-3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.out.exists():
        print(f"refusing to overwrite: {args.out}", file=sys.stderr)
        return 2

    m1 = json.loads(args.m1_request.read_text())
    listener_position, listener_wxyz = listener_pose_from_m1_request(m1)
    listener_yaw = listener_ue_yaw_deg(listener_wxyz)

    rows, mismatches = [], []
    max_gap = 0.0
    max_pos_drift = 0.0
    for point_dir in sorted(d for d in args.captures_root.iterdir()
                            if d.is_dir() and (d / "frame_records.json").is_file()):
        try:
            world, camera_yaw = captured_static_camera_world_m(point_dir)
        except (CurrentMP3DDynamicAudioError, OSError, json.JSONDecodeError) as exc:
            mismatches.append({"point_id": point_dir.name,
                               "error": f"{type(exc).__name__}: {exc}"})
            continue
        gap = circular_gap_deg(listener_yaw, camera_yaw)
        drift = max(abs(float(a) - float(b))
                    for a, b in zip(world, listener_position))
        max_gap = max(max_gap, gap)
        max_pos_drift = max(max_pos_drift, drift)
        row = {"point_id": point_dir.name, "capture_ue_yaw_deg": camera_yaw,
               "yaw_gap_deg": gap, "position_drift_m": drift}
        rows.append(row)
        if gap > args.tolerance_deg or drift > 1.0e-6:
            mismatches.append(row)

    version_boundary = None
    if args.audio_root and args.assertion_epoch:
        before, after = [], []
        for d in sorted(x for x in args.audio_root.iterdir() if x.is_dir()):
            receipt = d / "research_receipt.json"
            if not receipt.is_file():
                continue
            (after if receipt.stat().st_mtime >= args.assertion_epoch
             else before).append(d.name)
        version_boundary = {
            "assertion_epoch_unix": args.assertion_epoch,
            "rendered_before_yaw_assertion": len(before),
            "rendered_with_yaw_assertion": len(after),
            "first_render_with_assertion": after[0] if after else None,
            "note": ("同一批输出跨了代码版本边界:断言之前的输出未经该"
                     "检查;本工具的批级扫描对全部捕获补做了同一比对。"
                     "整批不得作为同一版本的正式认证数据。"),
        }

    payload = {
        "schema": "qa_v3_listener_yaw_scan_v1",
        "captures_root": str(args.captures_root),
        "m1_request": str(args.m1_request),
        "m1_listener_ue_yaw_deg": listener_yaw,
        "tolerance_deg": args.tolerance_deg,
        "checked": len(rows),
        "max_yaw_gap_deg": max_gap,
        "max_position_drift_m": max_pos_drift,
        "mismatches": mismatches,
        "distinct_capture_yaws": sorted({r["capture_ue_yaw_deg"] for r in rows}),
        "code_version_boundary": version_boundary,
        "status": "research_candidate",
        "qualification_claim": False,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(json.dumps({"checked": len(rows), "max_yaw_gap_deg": max_gap,
                      "max_position_drift_m": max_pos_drift,
                      "mismatches": len(mismatches),
                      "distinct_capture_yaws": payload["distinct_capture_yaws"],
                      "version_boundary": version_boundary},
                     ensure_ascii=False))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
