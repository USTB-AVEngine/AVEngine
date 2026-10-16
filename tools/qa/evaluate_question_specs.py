#!/usr/bin/env python3
"""Evaluate registry-bound QuestionSpecs and render a standalone review page."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.question_spec import (  # noqa: E402
    CLAIM_BOUNDARY,
    evaluate_question_specs,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _status_label(status: str) -> str:
    return {"pass": "已通过", "rejected": "已拒绝", "unsupported": "暂不支持"}.get(
        status, status
    )


def _render_html(evaluations: list[dict[str, Any]], output: Path) -> None:
    counts = Counter(item["status"] for item in evaluations)
    cards = []
    for item in evaluations:
        requirements = item.get("scenario_requirements") or {}
        modalities = "".join(
            f'<span class="chip modality">{html.escape(str(value))}</span>'
            for value in requirements.get("required_modalities", [])
        )
        facts = "".join(
            f'<li><code>{html.escape(str(value))}</code></li>'
            for value in requirements.get("required_facts", [])
        )
        answer = item.get("answer")
        reason = item.get("reason")
        if answer:
            result_block = (
                '<div class="result answer"><span>答案</span>'
                f'<strong>{html.escape(str(answer.get("label_zh") or answer.get("value")))}</strong></div>'
            )
        else:
            result_block = (
                '<div class="result reason"><span>边界</span>'
                f'<strong>{html.escape(str((reason or {}).get("code", "unknown")))}</strong>'
                f'<p>{html.escape(str((reason or {}).get("detail", "")))}</p></div>'
            )
        question = item.get("question") or "当前 Facts 不足，因此没有生成具体问题/答案。"
        cards.append(
            f"""
            <article class="card {html.escape(item['status'])}">
              <div class="card-head">
                <div><span class="spec-id">{html.escape(str(item['spec_id']))}</span>
                <h2>{html.escape(str(item['question_type_name_zh']))}</h2></div>
                <span class="badge">{_status_label(item['status'])}</span>
              </div>
              <p class="question">{html.escape(str(question))}</p>
              {result_block}
              <div class="modalities">{modalities}</div>
              <details><summary>所需 Facts</summary><ul>{facts}</ul></details>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>A 线 QuestionSpec Canary</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#0d1a2b; --text:#edf5ff;
      --muted:#93a7c0; --line:#223650; --pass:#36d399; --reject:#fb7185; --unsupported:#fbbf24; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-height:100vh; color:var(--text); background:
      radial-gradient(circle at 12% 0%,#12345b 0,transparent 33%),
      radial-gradient(circle at 95% 18%,#25325b 0,transparent 31%),var(--bg);
      font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1180px,calc(100% - 32px)); margin:auto; padding:52px 0 72px; }}
    .eyebrow {{ color:#7dd3fc; letter-spacing:.13em; text-transform:uppercase; font-weight:700; }}
    h1 {{ margin:.25rem 0 .7rem; font-size:clamp(2rem,5vw,4.25rem); line-height:1.05; letter-spacing:-.045em; }}
    .lead {{ max-width:780px; color:var(--muted); font-size:1.08rem; }}
    .summary {{ display:flex; gap:12px; flex-wrap:wrap; margin:28px 0 34px; }}
    .metric {{ min-width:135px; padding:14px 16px; border:1px solid var(--line); border-radius:16px;
      background:rgba(13,26,43,.76); backdrop-filter:blur(10px); }}
    .metric strong {{ display:block; font-size:1.55rem; }} .metric span {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:16px; }}
    .card {{ position:relative; overflow:hidden; padding:22px; border:1px solid var(--line); border-radius:20px;
      background:linear-gradient(160deg,rgba(18,36,58,.96),rgba(9,22,39,.96)); box-shadow:0 18px 48px rgba(0,0,0,.2); }}
    .card::before {{ content:""; position:absolute; inset:0 auto 0 0; width:4px; background:var(--pass); }}
    .card.rejected::before {{ background:var(--reject); }} .card.unsupported::before {{ background:var(--unsupported); }}
    .card-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:14px; }}
    .spec-id {{ color:#8fb4dc; font:700 .78rem ui-monospace,monospace; letter-spacing:.09em; }}
    h2 {{ margin:.15rem 0 0; font-size:1.28rem; }}
    .badge {{ border:1px solid color-mix(in srgb,var(--pass) 50%,transparent); color:var(--pass);
      padding:4px 9px; border-radius:999px; font-size:.78rem; white-space:nowrap; }}
    .rejected .badge {{ color:var(--reject); border-color:var(--reject); }}
    .unsupported .badge {{ color:var(--unsupported); border-color:var(--unsupported); }}
    .question {{ min-height:48px; margin:18px 0 14px; font-size:1.02rem; }}
    .result {{ padding:13px 14px; border:1px solid var(--line); border-radius:14px; background:#071321; }}
    .result span {{ color:var(--muted); display:block; font-size:.75rem; }}
    .result strong {{ font-size:1.06rem; }} .result p {{ color:var(--muted); margin:.35rem 0 0; font-size:.85rem; }}
    .modalities {{ display:flex; flex-wrap:wrap; gap:7px; margin:15px 0 8px; }}
    .chip {{ border-radius:999px; padding:3px 9px; background:#183454; color:#bde3ff; font-size:.75rem; }}
    details {{ color:var(--muted); margin-top:10px; }} summary {{ cursor:pointer; }}
    ul {{ margin:.55rem 0 0; padding-left:1.25rem; }} code {{ color:#b7cae2; font-size:.78rem; overflow-wrap:anywhere; }}
    footer {{ margin-top:30px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); }}
  </style>
</head>
<body><main>
  <div class="eyebrow">AVBench · Lead A · Executable Canary</div>
  <h1>QuestionSpec 不是自由出题器</h1>
  <p class="lead">它只引用受控人物/动物资产与声音库，先给出场景要求，再查询同一份 Facts；证据缺失、答案不唯一或当前能力不支持时，直接拒绝。</p>
  <section class="summary">
    <div class="metric"><strong>{len(evaluations)}</strong><span>固定问题类别</span></div>
    <div class="metric"><strong>{counts['pass']}</strong><span>当前真实通过</span></div>
    <div class="metric"><strong>{counts['rejected']}</strong><span>本组被拒绝</span></div>
    <div class="metric"><strong>{counts['unsupported']}</strong><span>明确缺口</span></div>
  </section>
  <section class="grid">{''.join(cards)}</section>
  <footer>{html.escape(CLAIM_BOUNDARY)}。遮挡者身份仍需 occluder instance segmentation，不能由 target-only 遮挡比例反推。</footer>
</main></body></html>"""
    output.write_text(document, encoding="utf-8")


def build(
    *,
    specs_path: Path,
    facts_path: Path,
    asset_registry_path: Path,
    sound_registry_path: Path,
    bindings_path: Path,
    expected_status_path: Path,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    specs = _read_json(specs_path)
    evaluations = evaluate_question_specs(
        specs,
        facts=_read_json(facts_path),
        asset_registry=_read_json(asset_registry_path),
        sound_registry=_read_json(sound_registry_path),
        event_sound_bindings=_read_json(bindings_path),
    )
    expected = _read_json(expected_status_path)
    actual = {item["spec_id"]: item["status"] for item in evaluations}
    if actual != expected:
        raise RuntimeError(f"QuestionSpec status mismatch: expected {expected}, got {actual}")
    counts = Counter(actual.values())
    _write_json(output / "evaluations.json", evaluations)
    _render_html(evaluations, output / "question_cards.html")
    manifest = {
        "schema": "avengine_lead_a_question_spec_canary_v1",
        "status": "pass",
        "claim_boundary": CLAIM_BOUNDARY,
        "question_count": len(evaluations),
        "status_counts": dict(sorted(counts.items())),
        "expected_status_by_spec": expected,
        "inputs": {
            "specs": str(specs_path),
            "facts": str(facts_path),
            "asset_registry": str(asset_registry_path),
            "sound_registry": str(sound_registry_path),
            "event_sound_bindings": str(bindings_path),
        },
        "artifacts": {
            "evaluations": "evaluations.json",
            "review_html": "question_cards.html",
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", required=True, type=Path)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--asset-registry", required=True, type=Path)
    parser.add_argument("--sound-registry", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--expected-status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(
        specs_path=args.specs.resolve(),
        facts_path=args.facts.resolve(),
        asset_registry_path=args.asset_registry.resolve(),
        sound_registry_path=args.sound_registry.resolve(),
        bindings_path=args.bindings.resolve(),
        expected_status_path=args.expected_status.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
