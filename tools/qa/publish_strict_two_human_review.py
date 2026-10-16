#!/usr/bin/env python3
"""Publish a lightweight, server-linked review for the strict two-human gates."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageOps

REPOSITORY = Path(__file__).resolve().parents[2]
PLAN_DEFAULT = REPOSITORY / "examples/qa/native_strict_two_human_publication_v1.json"
EXPANSION_ROOT = REPOSITORY / "tmp/lead_d_strict_two_human_expansion_v1"
CANARY_ROOT = REPOSITORY / "tmp/lead_d_strict_two_human_canary_v1"
REPORT_ROOT = REPOSITORY / "reports/lead_a"
DELIVERY_SCHEMA = "avengine_native_strict_two_human_sparse_review_delivery_v1"
DEFAULT_MEDIA_BASE_URL = "http://127.0.0.1:18765"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {role}: {resolved}")
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY / path).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _plan_validator() -> Any:
    path = Path(__file__).with_name("validate_strict_two_human_publication_plan.py")
    spec = importlib.util.spec_from_file_location("strict8_publication_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load publication-plan validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _media_url(path: Path, base_url: str) -> str:
    resolved = path.resolve()
    roots = (
        (CANARY_ROOT.resolve(), "a_strict8_canary"),
        (EXPANSION_ROOT.resolve(), "a_strict8"),
        (REPORT_ROOT.resolve(), "a_strict8_reports"),
    )
    for root, mount in roots:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        encoded = "/".join(quote(part) for part in relative.parts)
        return f"{base_url.rstrip('/')}/files/{quote(mount)}/{encoded}"
    raise RuntimeError(f"No read-only media mount for {resolved}")


def _acoustic_records(row: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    index = int(row["row_index"])
    if index == 1:
        plan = CANARY_ROOT / "exact_rir_plan_v3/rir_job_plan.json"
        cache = CANARY_ROOT / "exact_rir_cache_v4/receipt.json"
        delivery = CANARY_ROOT / "binaural_v5/delivery.json"
    elif index == 7:
        root = EXPANSION_ROOT / "row7_v2_acoustic_v1" / row["row_id"]
        plan = root / "exact_rir_plan_v1/rir_job_plan.json"
        cache = root / "rir_cache_v1/receipt.json"
        delivery = root / "binaural_v1/delivery.json"
    else:
        root = EXPANSION_ROOT / "acoustic_batch_v1" / row["row_id"]
        plan = root / "exact_rir_plan_v1/rir_job_plan.json"
        cache = root / "rir_cache_v1/receipt.json"
        delivery = root / "binaural_v1/delivery.json"
    audio = _resolve(str(capture["audio"]["authoritative_wav"]))
    plan_data = _load(plan)
    cache_data = _load(cache)
    delivery_data = _load(delivery)
    _require(len(plan_data.get("jobs", [])) == 2, f"{row['row_id']} RIR job count drift")
    _require(
        cache_data.get("status") == "pass"
        and cache_data.get("full_plan_complete") is True
        and cache_data.get("selected_job_count") == 2,
        f"{row['row_id']} RIR cache drift",
    )
    _require(
        delivery_data.get("status") == "pass"
        and delivery_data.get("qualification_claim") is False
        and delivery_data.get("episode_count") == 1,
        f"{row['row_id']} binaural delivery drift",
    )
    return {
        "exact_rir_plan": _record(plan, "exact_rir_plan"),
        "rir_cache_receipt": _record(cache, "rir_cache_receipt"),
        "binaural_delivery": _record(delivery, "binaural_delivery"),
        "binaural_wav": _record(audio, "binaural_wav"),
    }


def _row_delivery(row: dict[str, Any], plan: dict[str, Any], base_url: str) -> dict[str, Any]:
    capture_root = _resolve(str(row["capture_root"]))
    manifest_path = capture_root / "manifest.json"
    manifest = _load(manifest_path)
    visibility_path = capture_root / "pixel_visibility_truth.json"
    visibility = _load(visibility_path)
    target = visibility["per_instance"]["source1"]["frames"][0]
    distractor = visibility["per_instance"]["source2"]["frames"][0]
    rgb = capture_root / "rgb_frames/frame_000000.png"
    visual_video = capture_root / "native_rgb_visual_only.mp4"
    av_video = capture_root / "native_rgb_binaural.mp4"
    acoustics = _acoustic_records(row, manifest)
    gate_path = _resolve(str(row["cpu_gate"]))
    identity_catalog = plan["identity_catalog"]
    media = {
        "rgb": _record(rgb, "normal_rgb_f15"),
        "visual_video": _record(visual_video, "single_frame_visual_video"),
        "av_video": _record(av_video, "single_frame_binaural_video"),
        "audio": acoustics["binaural_wav"],
    }
    for record in media.values():
        record["url"] = _media_url(Path(record["path"]), base_url)
    evidence = {
        "capture_manifest": _record(manifest_path, "capture_manifest"),
        "pixel_visibility_truth": _record(visibility_path, "pixel_visibility_truth"),
        "metric_depth": _record(capture_root / "metric_depth_native.npz", "metric_depth"),
        "target_only_masks": _record(
            capture_root / "native_pixel_masks_depth_authority_v1.npz",
            "target_only_masks",
        ),
        "runtime_readbacks": _record(
            capture_root / "runtime_readbacks.json", "runtime_readbacks"
        ),
        "runtime_asset_readbacks": _record(
            capture_root / "runtime_asset_readbacks.json", "runtime_asset_readbacks"
        ),
        "cpu_gate": _record(gate_path, row["cpu_gate_kind"]),
        "exact_rir_plan": acoustics["exact_rir_plan"],
        "rir_cache_receipt": acoustics["rir_cache_receipt"],
        "binaural_delivery": acoustics["binaural_delivery"],
    }
    for record in evidence.values():
        record["url"] = _media_url(Path(record["path"]), base_url)
    target_x = float(target["target_centroid_xy_px"][0]) / 1280.0
    distractor_x = float(distractor["target_centroid_xy_px"][0]) / 1280.0
    return {
        "row_index": row["row_index"],
        "row_id": row["row_id"],
        "attempt_id": "v2" if row["row_index"] == 7 else "v1",
        "episode_id": row["episode_id"],
        "status": "pass_sparse_f15",
        "captured_frame_indices": [15],
        "formal_scene_count": 0,
        "qualification_claim": False,
        "target": {
            "identity_key": row["target_identity_key"],
            "identity_id": identity_catalog[row["target_identity_key"]],
            "side": row["target_side"],
            "sound_asset_id": row["target_sound_asset_id"],
            "speech_frame_window_inclusive": row["speech_frame_window_inclusive"],
            "visible_pixels": target["visible_pixels"],
            "visible_fraction": target["visible_fraction"],
            "occlusion_fraction": target["occlusion_fraction"],
            "centroid_x_fraction": target_x,
            "bbox_xyxy_px": target["target_bbox_xyxy_px"],
        },
        "distractor": {
            "identity_key": row["distractor_identity_key"],
            "identity_id": identity_catalog[row["distractor_identity_key"]],
            "side": "right" if row["target_side"] == "left" else "left",
            "speech_event_count": 0,
            "visible_pixels": distractor["visible_pixels"],
            "visible_fraction": distractor["visible_fraction"],
            "occlusion_fraction": distractor["occlusion_fraction"],
            "centroid_x_fraction": distractor_x,
            "bbox_xyxy_px": distractor["target_bbox_xyxy_px"],
        },
        "media": media,
        "evidence": evidence,
        "capture_process_exit_boundary": (
            "persisted_exit_zero" if row["row_index"] == 1 else "manifest_pass_no_persistent_receipt"
        ),
    }


def build_ledger(plan_path: Path, media_base_url: str) -> dict[str, Any]:
    validation = _plan_validator().validate(plan_path.resolve())
    plan = _load(plan_path.resolve())
    rows = [_row_delivery(row, plan, media_base_url) for row in plan["rows"]]
    rejection_plan = plan["excluded_attempts"][0]
    rejection = _load(_resolve(rejection_plan["rejection_record"]))
    rejected_rgb = _resolve(rejection_plan["capture_root"]) / "rgb_frames/frame_000000.png"
    history_rgb = _record(rejected_rgb, "row7_v1_rejected_rgb")
    history_rgb["url"] = _media_url(rejected_rgb, media_base_url)
    rejection_record = _record(
        _resolve(rejection_plan["rejection_record"]), "row7_v1_rejection_record"
    )
    rejection_record["url"] = _media_url(Path(rejection_record["path"]), media_base_url)
    return {
        "schema": DELIVERY_SCHEMA,
        "status": "pass",
        "title": "Strict Two-Human Native Sparse Review",
        "claim_boundary": plan["claim_boundary"],
        "counted_sparse_scene_count": 8,
        "sparse_pass_count": 8,
        "formal_scene_count": 0,
        "qualification_claim": False,
        "captured_frame_indices": [15],
        "target_side_counts": validation["target_side_counts"],
        "target_identity_counts": validation["target_identity_counts"],
        "distractor_identity_counts": validation["distractor_identity_counts"],
        "visibility_contract": plan["visibility_contract"],
        "review_boundaries": plan["review_boundaries"],
        "rows": rows,
        "excluded_attempts": [
            {
                "row_id": rejection_plan["row_id"],
                "attempt_id": "v1",
                "status": "rejected",
                "counted": False,
                "reason": rejection_plan["reason"],
                "observed_target_visible_fraction": rejection["target_gate"][
                    "observed_visible_fraction"
                ],
                "formal_scene_count": 0,
                "qualification_claim": False,
                "rejection_record": rejection_record,
                "rgb": history_rgb,
            }
        ],
    }


def _contact_sheet(ledger: dict[str, Any], output: Path) -> None:
    tile_width, image_height, label_height = 520, 292, 54
    canvas = Image.new("RGB", (tile_width * 2, (image_height + label_height) * 4), "#07111d")
    draw = ImageDraw.Draw(canvas)
    for offset, row in enumerate(ledger["rows"]):
        image = Image.open(row["media"]["rgb"]["path"]).convert("RGB")
        fitted = ImageOps.fit(image, (tile_width, image_height), method=Image.Resampling.LANCZOS)
        x = (offset % 2) * tile_width
        y = (offset // 2) * (image_height + label_height)
        canvas.paste(fitted, (x, y))
        draw.rectangle((x, y + image_height, x + tile_width, y + image_height + label_height), fill="#0d2133")
        label = (
            f"{row['row_index']:02d}  {row['target']['identity_key']} target {row['target']['side']}"
            f"  vis={row['target']['visible_fraction']:.3f}"
        )
        draw.text((x + 16, y + image_height + 16), label, fill="#dcfce7")
    canvas.save(output, format="PNG", optimize=True)


def _link(record: dict[str, Any], label: str) -> str:
    return (
        f'<a href="{html.escape(record["url"])}" target="_blank" rel="noreferrer">'
        f"{html.escape(label)} ↗</a>"
    )


def _html(ledger: dict[str, Any]) -> str:
    cards: list[str] = []
    for row in ledger["rows"]:
        target = row["target"]
        distractor = row["distractor"]
        media = row["media"]
        evidence = row["evidence"]
        occlusion = float(target["occlusion_fraction"]) * 100.0
        environment_note = (
            f'<span class="warn">environment occlusion {occlusion:.1f}%</span>'
            if occlusion > 0.0
            else '<span class="ok">clear</span>'
        )
        cards.append(
            f"""
            <article class="row-card" id="row-{row['row_index']}" data-counted-row>
              <div class="card-head"><div><span class="ordinal">{row['row_index']:02d}</span>
                <h3>{html.escape(target['identity_key'])} target · {html.escape(target['side'])}</h3></div>
                <span class="pass">PASS sparse f15</span></div>
              <img src="{html.escape(media['rgb']['url'])}" alt="Row {row['row_index']} native RGB frame 15">
              <div class="metrics">
                <div><small>target visibility</small><strong>{target['visible_fraction']:.3f}</strong></div>
                <div><small>distractor visibility</small><strong>{distractor['visible_fraction']:.3f}</strong></div>
                <div><small>target pixels</small><strong>{target['visible_pixels']:,}</strong></div>
                <div><small>target state</small><strong>{environment_note}</strong></div>
              </div>
              <p class="identity">Target <b>{html.escape(target['identity_id'])}</b><br>
                Silent distractor <b>{html.escape(distractor['identity_id'])}</b></p>
              <p class="speech">{html.escape(target['sound_asset_id'])} · frames
                {target['speech_frame_window_inclusive'][0]}–{target['speech_frame_window_inclusive'][1]}</p>
              <details><summary>Single-frame video and 5 s binaural audio</summary>
                <video controls preload="metadata" src="{html.escape(media['av_video']['url'])}"></video>
                <audio controls preload="metadata" src="{html.escape(media['audio']['url'])}"></audio>
                <p class="boundary">The MP4 contains one sparse frame; the 5 s WAV is the authoritative listening artifact.</p>
              </details>
              <div class="links">{_link(media['visual_video'], 'visual MP4')}
                {_link(media['av_video'], 'AV MP4')} {_link(media['audio'], 'binaural WAV')}
                {_link(evidence['pixel_visibility_truth'], 'pixel truth')}
                {_link(evidence['metric_depth'], 'metric depth')}
                {_link(evidence['runtime_asset_readbacks'], 'live6')}
                {_link(evidence['exact_rir_plan'], 'RIR plan')}
                {_link(evidence['rir_cache_receipt'], 'RIR receipt')}</div>
              <p class="receipt">Capture receipt: {html.escape(row['capture_process_exit_boundary'])}</p>
            </article>"""
        )
    rejected = ledger["excluded_attempts"][0]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src http://127.0.0.1:18765 data:; media-src http://127.0.0.1:18765; connect-src http://127.0.0.1:18765; base-uri 'none'; form-action 'none'">
<title>Strict Two-Human · Native Sparse Review</title><style>
:root{{--bg:#06101a;--panel:#0a1928;--line:#1c3850;--text:#e9f4fb;--muted:#92abc0;--mint:#5ee6b5;--cyan:#54c6ff;--amber:#ffc66d;--red:#ff7b8b}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 20% -10%,#123955 0,transparent 38%),var(--bg);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{max-width:1480px;margin:auto;padding:48px 28px 80px}}h1{{font-size:clamp(38px,6vw,76px);line-height:1;margin:.2em 0}}h2{{font-size:28px;margin:48px 0 18px}}h3{{margin:0;font-size:20px}}.eyebrow{{text-transform:uppercase;letter-spacing:.18em;color:var(--mint);font-weight:800}}.lede{{max-width:900px;color:#b8cddd;font-size:18px}}.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}}.summary div,.boundary-box{{background:#091a29;border:1px solid var(--line);border-radius:18px;padding:20px}}.summary strong{{display:block;font-size:34px;color:var(--mint)}}.summary small,.muted,.receipt{{color:var(--muted)}}.contact{{width:100%;border:1px solid var(--line);border-radius:20px;display:block}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.row-card{{background:linear-gradient(150deg,#0c1f30,#081622);border:1px solid var(--line);border-radius:22px;padding:18px;box-shadow:0 18px 50px #0005}}.card-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}}.card-head>div{{display:flex;gap:12px;align-items:center}}.ordinal{{display:grid;place-items:center;width:40px;height:40px;border-radius:12px;background:#12334a;color:var(--cyan);font-weight:900}}.pass{{color:var(--mint);background:#0d3c32;padding:6px 10px;border-radius:999px;font-weight:800}}.row-card>img{{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:14px;background:#000}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:14px 0}}.metrics div{{background:#07131e;border-radius:12px;padding:10px}}.metrics small{{color:var(--muted);display:block}}.metrics strong{{font-size:17px}}.ok{{color:var(--mint)}}.warn{{color:var(--amber)}}.identity,.speech,.boundary{{color:#b9cede}}details{{border:1px solid var(--line);padding:10px 12px;border-radius:12px}}summary{{cursor:pointer;color:var(--cyan);font-weight:700}}video,audio{{width:100%;margin-top:12px}}.links{{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}}a{{color:#8bd9ff;text-decoration:none;background:#10283a;border:1px solid #22435d;padding:6px 9px;border-radius:9px}}.rejected{{border:1px solid #6b2c39;background:#211018;border-radius:20px;padding:18px;display:grid;grid-template-columns:320px 1fr;gap:18px}}.rejected img{{width:100%;border-radius:12px}}.rejected h3{{color:var(--red)}}code{{color:#d1e8f7}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}.summary{{grid-template-columns:repeat(2,1fr)}}.metrics{{grid-template-columns:repeat(2,1fr)}}.rejected{{grid-template-columns:1fr}}}}@media(max-width:520px){{main{{padding:28px 14px}}.summary{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="eyebrow">A × D · controlled native evidence</p><h1>Strict Two‑Human<br>Native Sparse Review</h1>
<p class="lede">Eight distinct adult–adult Apartment configurations passed the current f15 native gate. Every counted row closes normal RGB, metric depth, two target-only passes, live UE asset readback, exact two-endpoint RIR, and a 5-second binaural mix.</p>
<section class="summary"><div><strong>8 / 8</strong><small>counted sparse gates</small></div><div><strong>4 + 4</strong><small>target left / right</small></div><div><strong>3</strong><small>original adult identities</small></div><div><strong>0</strong><small>formal scenes</small></div></section>
<div class="boundary-box"><b>Claim boundary.</b> These are single-frame f15 research gates, not full 75-frame Facts, formal benchmark scenes, or qualification evidence. Target-only RGB was not persisted; ground contact is not claimed. Rows 2–8 have capture-manifest PASS but no standalone persistent process-exit receipt.</div>
<h2>At a glance</h2><img class="contact" src="contact_sheet.png" alt="Eight accepted strict two-human frames">
<h2>Counted rows</h2><section class="grid">{''.join(cards)}</section>
<h2>Rejected history · excluded from denominator</h2><section class="rejected"><img src="{html.escape(rejected['rgb']['url'])}" alt="Rejected row 7 v1"><div><h3>Row 07 · v1 rejected</h3><p>{html.escape(rejected['reason'])}</p><p>Observed target visibility <b>{rejected['observed_target_visible_fraction']:.3f}</b>, below the current <b>0.800</b> gate. The corrected v2 is the counted row above.</p>{_link(rejected['rejection_record'], 'rejection record')}</div></section>
<h2>Reading notes</h2><div class="boundary-box"><p><b>Visual authority:</b> original normal RGB plus metric-depth-derived target-only masks. The target-only passes do not persist standalone RGB images.</p><p><b>Audio authority:</b> linked 5 s binaural WAV; the sparse MP4 is only a one-frame inspection wrapper.</p><p><b>Runtime authority:</b> stable tag, exact Blueprint, mesh, skeleton, Standing Idle, and actor-root plus declared emitter offset are live-read back for both people.</p><p><b>Not claimed:</b> foot/socket floor contact, full-episode pixel truth, formal admission, or benchmark qualification.</p></div>
</main></body></html>"""


def _verify_record(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    _require(path.is_file(), f"Published source missing: {path}")
    _require(path.stat().st_size == record["size_bytes"], f"Published source size drift: {path}")
    _require(_sha256(path) == record["sha256"], f"Published source hash drift: {path}")


def verify(output: Path) -> dict[str, Any]:
    output = output.resolve()
    ledger = _load(output / "ledger.json")
    _require(ledger.get("schema") == DELIVERY_SCHEMA, "delivery schema drift")
    _require(
        ledger.get("status") == "pass"
        and ledger.get("sparse_pass_count") == 8
        and ledger.get("formal_scene_count") == 0
        and ledger.get("qualification_claim") is False,
        "delivery claim boundary drift",
    )
    _require(len(ledger.get("rows", [])) == 8, "delivery row count drift")
    _require(Counter(row["target"]["side"] for row in ledger["rows"]) == {"left": 4, "right": 4}, "delivery side drift")
    for row in ledger["rows"]:
        for record in row["media"].values():
            _verify_record(record)
        for record in row["evidence"].values():
            _verify_record(record)
    rejected = ledger["excluded_attempts"]
    _require(len(rejected) == 1 and rejected[0]["counted"] is False, "rejected history drift")
    _verify_record(rejected[0]["rejection_record"])
    _verify_record(rejected[0]["rgb"])
    html_text = (output / "index.html").read_text(encoding="utf-8")
    _require(html_text.count("data-counted-row") == 8, "HTML row count drift")
    _require("formal scenes</small>" in html_text and ">0<" in html_text, "HTML formal boundary missing")
    _require((output / "contact_sheet.png").is_file(), "contact sheet missing")
    return {
        "schema": "avengine_native_strict_two_human_sparse_review_verification_v1",
        "status": "pass",
        "counted_sparse_scene_count": 8,
        "formal_scene_count": 0,
        "qualification_claim": False,
    }


def publish(plan: Path, output: Path, media_base_url: str) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to clobber publication: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger = build_ledger(plan.resolve(), media_base_url)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        _write(staging / "ledger.json", ledger)
        _contact_sheet(ledger, staging / "contact_sheet.png")
        (staging / "index.html").write_text(_html(ledger), encoding="utf-8")
        verify(staging)
        os.replace(staging, output)
        return verify(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--plan", type=Path, default=PLAN_DEFAULT)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--plan", type=Path, default=PLAN_DEFAULT)
    publish_parser.add_argument("--output", type=Path, required=True)
    publish_parser.add_argument("--media-base-url", default=DEFAULT_MEDIA_BASE_URL)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        result = _plan_validator().validate(args.plan.resolve())
    elif args.command == "publish":
        result = publish(args.plan, args.output, args.media_base_url)
    else:
        result = verify(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
