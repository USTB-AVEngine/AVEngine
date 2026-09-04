#!/usr/bin/env python3
"""Positive control for the visibility predictor: table prediction vs pixel truth.

For every captured card1 candidate (a pixel directory with evidence.json and
pixel_visibility_truth.json) the tool reconstructs the camera pose and each
actor's position at every captured frame from the final timeline, predicts
the visible fraction from the camera clearance table, and compares it with
the renderer's own per-frame visibility (state, visible_fraction).

It reports per-frame agreement (correlation, absolute error), tier agreement
under the pixel join ladder, and the hidden/visible and out-of-view confusion
tables, grouped by scene.  The predictor is not admitted for rejection until
this comparison is judged adequate; until then it only budgets tiers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools" / "qa"))

from camera_clearance import CameraClearanceError, CameraClearanceTable, point_key  # noqa: E402
from qa_v3_pixel_thresholds import tier_for_frame  # noqa: E402
from visibility_prediction import (  # noqa: E402
    TIER_EDGES_DEFAULT,
    body_from_params,
    predict_point_visibility,
    predicted_tier,
    relative_azimuth,
)

SCHEMA = "qa_v3_visibility_prediction_validation_v1"


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidate_rows(pixel_dir: Path, table: CameraClearanceTable, *, body: dict,
                   hfov_deg: float, ground_z_cm: float, edges: Sequence[float]
                   ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence = _read(pixel_dir / "evidence.json")
    truth = _read(pixel_dir / "pixel_visibility_truth.json")
    timeline = _read(evidence["inputs"]["timeline"])
    frames = {int(f["frame_index"]): f for f in timeline["frames"]}
    camera0 = frames[min(frames)]["camera"]
    camera_xy = (float(camera0["translation_ue_cm"][0]), float(camera0["translation_ue_cm"][1]))
    camera_height_m = (float(camera0["translation_ue_cm"][2]) - ground_z_cm) / 100.0
    yaw = float(camera0["yaw_ue_deg"])
    slots = sorted({s["source_slot_id"] for s in frames[min(frames)]["actor_states"]})
    per_instance = truth["per_instance"]
    rows = []
    for key, instance in per_instance.items():
        slot = key if key in slots else next(
            (s for s in slots if s in key or key in s), None)
        if slot is None:
            continue
        for record in instance["frames"]:
            frame = int(record["frame_index"])
            if frame not in frames:
                continue
            states = {s["source_slot_id"]: s for s in frames[frame]["actor_states"]}
            actor = states[slot]["translation_ue_cm"][:2]
            others = [(states[o]["translation_ue_cm"][:2], body) for o in slots if o != slot]
            prediction = predict_point_visibility(
                table, camera_xy_cm=camera_xy, camera_height_m=camera_height_m,
                ground_z_cm=ground_z_cm, actor_xy_cm=actor, body=body, others=others)
            azimuth = relative_azimuth(camera_xy, yaw, actor)
            in_fov = abs(azimuth) <= hfov_deg / 2.0
            truth_fraction = record.get("visible_fraction")
            truth_state = record.get("state")
            rows.append({
                "candidate": pixel_dir.name, "slot": slot, "frame": frame,
                "relative_azimuth_deg": round(azimuth, 3), "in_fov_geometric": in_fov,
                "truth_state": truth_state,
                "truth_visible_fraction": (None if truth_fraction is None
                                           else float(truth_fraction)),
                "truth_tier": tier_for_frame(truth_state, truth_fraction, edges),
                "predicted_visible_fraction": prediction["predicted_visible_fraction"],
                "predicted_tier": predicted_tier(
                    in_fov, prediction["predicted_visible_fraction"], edges),
                "known_fraction": prediction["known_fraction"],
                "blocked_by_scene": prediction["blocked_by_scene"],
                "blocked_by_actor": prediction["blocked_by_actor"],
                "distance_m": prediction["distance_m"],
            })
    meta = {"candidate": pixel_dir.name, "timeline": evidence["inputs"]["timeline"],
            "camera_key": point_key(camera_xy), "camera_height_m": camera_height_m,
            "camera_yaw_deg": yaw, "rows": len(rows)}
    return rows, meta


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    in_view = [r for r in rows if r["truth_state"] != "out_of_view" and r["in_fov_geometric"]
               and r["predicted_visible_fraction"] is not None
               and r["truth_visible_fraction"] is not None]
    pred = np.asarray([r["predicted_visible_fraction"] for r in in_view], dtype=float)
    true = np.asarray([r["truth_visible_fraction"] for r in in_view], dtype=float)
    out: dict[str, Any] = {"rows": len(rows), "rows_in_view_both": len(in_view)}
    if len(in_view) >= 3:
        out["pearson"] = float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 and true.std() > 0 else None
        rank_p = np.argsort(np.argsort(pred)); rank_t = np.argsort(np.argsort(true))
        out["spearman"] = float(np.corrcoef(rank_p, rank_t)[0, 1]) if pred.std() > 0 and true.std() > 0 else None
        out["abs_error_median"] = float(np.median(np.abs(pred - true)))
        out["abs_error_mean"] = float(np.mean(np.abs(pred - true)))
        hidden_true = true <= 0.0
        hidden_pred = pred <= 0.0
        out["hidden_confusion"] = {
            "true_hidden_pred_hidden": int((hidden_true & hidden_pred).sum()),
            "true_hidden_pred_visible": int((hidden_true & ~hidden_pred).sum()),
            "true_visible_pred_hidden": int((~hidden_true & hidden_pred).sum()),
            "true_visible_pred_visible": int((~hidden_true & ~hidden_pred).sum())}
        low_true = true < 0.5
        low_pred = pred < 0.5
        out["below_half_confusion"] = {
            "agree": int((low_true == low_pred).sum()),
            "disagree": int((low_true != low_pred).sum())}
    tiers = [(r["truth_tier"], r["predicted_tier"]) for r in rows]
    out["tier_agreement"] = sum(1 for t, p in tiers if t == p) / len(tiers) if tiers else None
    confusion: dict[str, dict[str, int]] = {}
    for t, p in tiers:
        confusion.setdefault(t, {})
        confusion[t][p] = confusion[t].get(p, 0) + 1
    out["tier_confusion_truth_to_predicted"] = confusion
    oov_true = [r["truth_state"] == "out_of_view" for r in rows]
    oov_pred = [not r["in_fov_geometric"] for r in rows]
    out["out_of_view_agreement"] = (sum(1 for a, b in zip(oov_true, oov_pred) if a == b)
                                    / len(rows)) if rows else None
    out["unknown_prediction_rows"] = sum(1 for r in rows if r["predicted_visible_fraction"] is None
                                         and r["in_fov_geometric"])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, type=Path,
                        help="camera clearance table directory for the scene")
    parser.add_argument("--scene-config", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--pixel-dir", action="append", required=True, type=Path,
                        help="captured candidate directory (evidence.json + pixel_visibility_truth.json)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    config = _read(args.scene_config)
    params = _read(args.params)
    from scene_sampler import scene_hfov_deg
    hfov = scene_hfov_deg(config)
    ground_z = float((config.get("render") or {})["ground_z_ue_cm"])
    edges = tuple(float(v) for v in params.get("PIXEL_TIER_VISIBLE_FRACTION_EDGES",
                                                 TIER_EDGES_DEFAULT))
    body = body_from_params(params)
    table = CameraClearanceTable.load(args.table)
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for pixel_dir in args.pixel_dir:
        try:
            candidate_rows_, meta = candidate_rows(
                pixel_dir, table, body=body, hfov_deg=hfov, ground_z_cm=ground_z, edges=edges)
        except CameraClearanceError as error:
            skipped.append({"candidate": pixel_dir.name, "reason": str(error)})
            continue
        rows.extend(candidate_rows_)
        candidates.append(meta)
    result = {
        "schema": SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": ("prediction vs renderer pixel truth on already captured "
                           "candidates; a positive control for the predictor, not "
                           "question admission"),
        "table": table.identity,
        "scene_id": config["scene_id"],
        "body": body,
        "tier_edges": list(edges),
        "candidates": candidates,
        "skipped": skipped,
        "summary": summarise(rows),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    summary = result["summary"]
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("tier_confusion_truth_to_predicted",)}, ensure_ascii=False))
    print(f"QA_V3_VISIBILITY_VALIDATION_OK output={args.output} candidates={len(candidates)} "
          f"skipped={len(skipped)} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
