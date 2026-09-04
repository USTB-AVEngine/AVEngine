#!/usr/bin/env python3
"""Batch-level verification of qa-v3 pilot audio renders (post-render gate).

对整个音频批做四层检查,任何一层失败即非零退出(失败即停):
1. 双声道结构:每个 mixture.wav 过 1.1 校验器同款判定(双声道 + 左右
   逐样本差异达标),这里直接内联同一判定式以免子进程 ×288;
2. receipt 一致性:endpoint registry 必须是 qa_v2 表、audio_program 路径
   必须指向该点自己的 program(孪生用派生 program)、qualification_claim
   恒 false;
3. onset 落位抽验:每点每事件,事件内 RMS 必须显著高于事件前静默段
   (比值下限显式参数);锚事件必须是最后事件且锚后尾静默 ≥ TAIL_MIN_S;
4. **音频变体非同一**:同点 main 与 gateA 的 mixture 波形最大逐样本差
   必须超过阈值。**这一层只证明音频确实被改了,不证明 Gate A 语义
   成立** —— 改增益、改噪声、改任意事件都能让波形不同。Gate A 要求的
   是"只改承载选择事实的音频变量、视觉不变、该题型的金标按预期翻转",
   那要逐题比对 main/gateA 的事实记录,不能由波形差异代替。

输出批级 manifest(no-clobber)。research_candidate。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

AMP_EPSILON = 1e-6
MIN_DIFF_RATIO = 0.01
ONSET_RMS_RATIO_MIN = 5.0     # 事件内/事件前 RMS 比值下限(显式参数)
GATEA_MAX_DIFF_MIN = 1e-4     # main vs gateA 最大逐样本差下限
SR = 16000


def check_stereo(wav: np.ndarray) -> str | None:
    if wav.ndim != 2 or wav.shape[1] != 2:
        return f"not stereo: shape={wav.shape}"
    diff = np.abs(wav[:, 0] - wav[:, 1])
    ratio = float((diff > AMP_EPSILON).mean())
    if ratio < MIN_DIFF_RATIO:
        return f"channels nearly identical: diff_ratio={ratio:.5f}"
    return None


def check_receipt(receipt: dict, expect_program: Path) -> str | None:
    if receipt.get("qualification_claim") is not False:
        return "qualification_claim is not false"
    reg = receipt["inputs"]["source_endpoint_registry"]["path"]
    if "qa_v2/source_endpoints_qa_v2_v1.json" not in reg:
        return f"wrong endpoint registry: {reg}"
    prog = receipt["audio_program"]["path"]
    if Path(prog).name != expect_program.name:
        return f"program mismatch: {Path(prog).name} != {expect_program.name}"
    return None


def find_program(programs_dir: Path, base: str, variant: str) -> Path | None:
    suffixes = (
        ["_rand_v1.json"] if variant == "main" else
        ["_rand_gateA_v1.json", "_gateA_rand_v1.json"]
    )
    matches = []
    for suffix in suffixes:
        matches.extend(programs_dir.glob(f"qa_v3_*_{base}{suffix}"))
    unique = sorted(set(matches))
    return unique[0] if len(unique) == 1 else None


def check_onsets(wav: np.ndarray, program: dict, authority: dict,
                 profile_id: str | None,
                 tail_min_s: float) -> list[str]:
    """Check event energy plus profile-specific anchor/tail structure.

    ``authority`` is a current fact record, or a retained run01 plan for
    backward-compatible verification of historical evidence.
    """
    errors = []
    mono = np.abs(wav).mean(axis=1)
    events = sorted(program["events"], key=lambda e: e["start_sample"])
    for e in events:
        s0, s1 = e["start_sample"], e["end_sample_exclusive"]
        pre = float(mono[max(0, s0 - 1600):s0].mean()) if s0 >= 800 else 0.0
        inside = float(mono[s0:min(s1, s0 + 4800)].mean())
        if inside <= 0:
            errors.append(f"event@{s0}: zero energy inside event")
        elif pre > 0 and inside / pre < ONSET_RMS_RATIO_MIN:
            errors.append(f"event@{s0}: inside/pre RMS {inside / pre:.2f} "
                          f"< {ONSET_RMS_RATIO_MIN}")
    if "anchor_end_sample" in authority:
        anchor_end = int(authority["anchor_end_sample"])
        if anchor_end != max(e["end_sample_exclusive"] for e in events):
            errors.append("anchor is not the last event")
        tail_s = (len(mono) - anchor_end) / SR
        if tail_s < tail_min_s - 1e-9:
            errors.append(f"tail {tail_s:.3f}s < TAIL_MIN {tail_min_s}")
        return errors
    identity_starts = {
        int(event["start_sample"])
        for event in authority.get("audio", {}).get("events", [])
        if event.get("purpose") == "identity_anchor"
    }
    identity = [
        event for event in events
        if int(event["start_sample"]) in identity_starts
    ]
    if profile_id in {"card1F", "card1B"}:
        if len(identity) != 1 or identity[0] is not events[-1]:
            errors.append("identity anchor is not the unique last event")
        elif profile_id == "card1F":
            tail_s = (len(mono) - identity[0]["end_sample_exclusive"]) / SR
            if tail_s < tail_min_s - 1e-9:
                errors.append(f"tail {tail_s:.3f}s < TAIL_MIN {tail_min_s}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design-root", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.out.exists():
        print(f"refusing to overwrite: {args.out}", file=sys.stderr)
        return 2
    params = json.loads(args.params.read_text())
    tail_min_s = float(params["TAIL_MIN_S"])
    programs_dir = args.design_root / "programs"

    point_dirs = sorted(d for d in args.audio_root.iterdir()
                        if d.is_dir() and (d / "research_receipt.json").is_file())
    failures: list[str] = []
    checked = gatea_pairs = gatea_semantic_pairs = 0
    for d in point_dirs:
        name = d.name
        base = name[:-6] if name.endswith("_gateA") else name
        variant = "gateA" if name.endswith("_gateA") else "main"
        prog_path = find_program(programs_dir, base, variant)
        if prog_path is None:
            failures.append(f"{name}: program lookup was not unique")
            continue
        program = json.loads(prog_path.read_text())
        fact_name = "fact_record_gateA.json" if variant == "gateA" else "fact_record.json"
        fact_path = args.design_root / base / fact_name
        fact = json.loads(fact_path.read_text()) if fact_path.is_file() else None
        plan_path = programs_dir / (
            prog_path.name.replace("_gateA_rand_v1.json", "_rand_v1.plan.json")
            .replace("_rand_gateA_v1.json", "_rand_v1.plan.json")
            if variant == "gateA" else prog_path.stem + ".plan.json"
        )
        plan = json.loads(plan_path.read_text()) if plan_path.is_file() else None
        try:
            wav, sr = sf.read(d / "audio" / "binaural" / "mixture.wav")
            receipt = json.loads((d / "research_receipt.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{name}: load failed: {exc}")
            continue
        if sr != SR or len(wav) != 80000:
            failures.append(f"{name}: wav {sr}Hz {len(wav)} samples")
            continue
        for err in filter(None, [check_stereo(wav),
                                 check_receipt(receipt, prog_path)]):
            failures.append(f"{name}: {err}")
        if variant == "main":
            authority = fact if fact is not None else plan
            if authority is None:
                failures.append(f"{name}: no fact or retained plan authority")
                authority = {}
            for err in check_onsets(
                wav, program, authority,
                fact.get("profile_id") if fact else None, tail_min_s,
            ):
                failures.append(f"{name}: {err}")
        checked += 1
        if variant == "gateA":
            main_wav_path = args.audio_root / base / "audio" / "binaural" / "mixture.wav"
            if main_wav_path.is_file():
                main_wav, _ = sf.read(main_wav_path)
                max_diff = float(np.abs(wav - main_wav).max())
                if max_diff < GATEA_MAX_DIFF_MIN:
                    failures.append(f"{name}: gateA identical to main "
                                    f"(max diff {max_diff:.2e})")
                gatea_pairs += 1
            main_fact_path = args.design_root / base / "fact_record.json"
            gate_fact_path = args.design_root / base / "fact_record_gateA.json"
            if main_fact_path.is_file() and gate_fact_path.is_file():
                main_fact = json.loads(main_fact_path.read_text())
                gate_fact = json.loads(gate_fact_path.read_text())
                checks = main_fact.get("gatea", {}).get("checks", {})
                required_checks = (
                    "event_count_same", "candidate_endpoints_same",
                    "non_slot_event_fields_same", "slot_sequence_changed",
                    "mcq_stem_same", "mcq_options_same", "open_stem_same",
                    "mcq_gold_flipped", "open_gold_separated",
                )
                missing = [key for key in required_checks
                           if checks.get(key) is not True]
                if missing:
                    failures.append(f"{name}: Gate A semantic checks failed {missing}")
                elif (
                    main_fact["mcq"]["truth_option"] == gate_fact["mcq"]["truth_option"]
                    or main_fact["open"]["truth_value"] == gate_fact["open"]["truth_value"]
                ):
                    failures.append(f"{name}: Gate A fact gold did not flip")
                else:
                    gatea_semantic_pairs += 1

    payload = {
        "schema": "qa_v3_audio_batch_verification_v1",
        "audio_root": str(args.audio_root),
        "checked_renders": checked,
        "audio_variant_waveform_nonidentity_pairs": gatea_pairs,
        "gatea_semantic_flip_pairs": gatea_semantic_pairs,
        "gatea_semantic_flip": (
            "established from current paired fact records and structural checks"
            if gatea_semantic_pairs else
            "not established by this tool for retained rows without paired facts"
        ),
        "onset_rms_ratio_min": ONSET_RMS_RATIO_MIN,
        "gatea_max_diff_min": GATEA_MAX_DIFF_MIN,
        "failures": failures,
        "status": "research_candidate",
        "qualification_claim": False,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"checked={checked} gateA_pairs={gatea_pairs} "
          f"gateA_semantic_pairs={gatea_semantic_pairs} "
          f"failures={len(failures)} out={args.out}")
    if failures:
        for f in failures[:12]:
            print("FAIL:", f, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
