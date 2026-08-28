"""Production surfaces: the sound-asset deck, the stage board, human verdicts.

Three rules shape everything here.

The asset catalog is read from the published tree on every request, not from a
cache and not from the index alone. Each asset's ``asset.json`` is the
authority for its own acceptance record - the index is rebuilt less often than
assets are re-measured, and a deck that shows last week's verdict next to this
week's mesh is worse than a slower page.

The board reads each stage's own verifier artifact and never a task's exit
code. Every product in this repository carries its own acceptance evidence -
a room manifest that validates, a package that hashes, a route report that
counts, an episode receipt whose per-frame errors are enumerated - and exit
code zero only says a process ended. A board painted from exit codes shows the
first silent failure as green.

Human verdicts are files inside the task directory they judge, one file per
task, written whole. No database: the queue directory already survives server
restarts, and a verdict that lives next to the artifacts it judges cannot
dangle.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class ProductionError(ValueError):
    pass


_ASSET_FILE_SUFFIXES = frozenset({".glb", ".json", ".png"})
_VERDICT_VALUES = frozenset({"pass", "fail", "unsure"})
_VERDICT_FILE = "human_verdict.json"

# One join key across every stage's artifacts. Stage outputs name the scene
# three different ways - room_id hm3d_val_00800_TEEsavR23oF, bank directory
# 00800-TEEsavR23oF, receipt scene_id TEEsavR23oF - and the 11-character
# Matterport hash is the part they all carry.
_SCENE_HASH = re.compile(r"([A-Za-z0-9]{11})(?:\.|$|_)")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


# --------------------------------------------------------------------------
# sound-asset deck


def load_sound_asset_catalog(index_path: Path) -> dict[str, Any]:
    root = index_path.parent
    index = _read_json(index_path) or {}
    entries: list[dict[str, Any]] = []
    for asset_json in sorted(root.glob("*/*/*/asset.json")):
        record = _read_json(asset_json)
        if record is None:
            continue
        relative = asset_json.parent.relative_to(root).as_posix()
        geometry = record.get("geometry") or {}
        acceptance = record.get("acceptance") or {}
        placement = record.get("placement") or {}
        resting = (geometry.get("resting_pose") or {}) if isinstance(
            geometry.get("resting_pose"), Mapping
        ) else {}
        files = {}
        for name in ("finalized.glb", "emitter_marker.glb"):
            candidate = asset_json.parent / name
            if candidate.is_file():
                files[name] = {
                    "path": f"{relative}/{name}",
                    "byte_size": candidate.stat().st_size,
                }
        entries.append(
            {
                "asset_id": record.get("asset_id"),
                "path": relative,
                "category": record.get("category"),
                "object_type": (record.get("identity") or {}).get("object_type"),
                "entity_class": record.get("entity_class"),
                "admission_state": record.get("admission_state"),
                "registration_authorized": record.get(
                    "formal_dataset_registration_authorized"
                ),
                "size_m": {
                    "width": geometry.get("width_right_m"),
                    "depth": geometry.get("depth_forward_m"),
                    "height": geometry.get("height_up_m"),
                },
                "resting_pose_verdict": acceptance.get("resting_pose_verdict"),
                "base_normal_tilt_deg": acceptance.get("base_normal_tilt_deg"),
                "mounting_plane_normal_tilt_deg": acceptance.get(
                    "mounting_plane_normal_tilt_deg"
                ),
                "attachment_surface": placement.get("attachment_surface")
                or resting.get("attachment_surface"),
                "attachment_surface_assumed": resting.get(
                    "attachment_surface_assumed"
                ),
                "has_emitter": isinstance(record.get("emitter"), Mapping),
                "files": files,
            }
        )
    return {
        "root": str(root),
        "index_created_at": index.get("created_at"),
        "asset_count": len(entries),
        "assets": entries,
    }


def sound_asset_file(index_path: Path, relative: str) -> Path:
    root = index_path.parent.resolve()
    target = (root / relative).resolve()
    if not str(target).startswith(str(root) + "/"):
        raise ProductionError("asset path escapes the asset root")
    if target.suffix not in _ASSET_FILE_SUFFIXES:
        raise ProductionError(f"not a servable asset file: {relative}")
    if not target.is_file():
        raise ProductionError(f"no such asset file: {relative}")
    return target


# --------------------------------------------------------------------------
# the dry-sound library

_SOUND_FILE_SUFFIXES = frozenset({".wav", ".json"})


def _wav_facts(path: Path) -> dict[str, Any]:
    """Header facts a curator needs before auditioning anything.

    wave covers the PCM files this pipeline consumes; a file it cannot parse
    is reported as such rather than hidden, because a clip that cannot even
    state its sample rate is itself a finding.
    """

    import contextlib
    import wave

    try:
        with contextlib.closing(wave.open(str(path), "rb")) as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return {
                "sample_rate_hz": rate,
                "channel_count": handle.getnchannels(),
                "sample_count": frames,
                "duration_s": round(frames / rate, 3) if rate else None,
                "readable": True,
            }
    except (wave.Error, EOFError, OSError) as error:
        return {"readable": False, "error": f"{type(error).__name__}: {error}"}


def load_sound_library(
    library_root: Path, asset_index_path: Path | None
) -> dict[str, Any]:
    """Dry-clip supply against the asset tree's own demand, read live.

    Demand is not invented here: it is the union of every published asset's
    allowed_event_classes and default_event_class - the vocabulary the audio
    programs actually select by. Supply is whatever wav files sit under the
    library root, each with an optional sidecar json declaring which event
    classes it claims to serve, its source and its licence. The page exists
    to make the gap between the two columns impossible to miss.
    """

    demand: dict[str, int] = {}
    if asset_index_path is not None:
        for asset_json in sorted(asset_index_path.parent.glob("*/*/*/asset.json")):
            record = _read_json(asset_json) or {}
            profile = record.get("acoustic_profile") or {}
            names = set(profile.get("allowed_event_classes") or [])
            default = profile.get("default_event_class")
            if default:
                names.add(str(default))
            for name in names:
                demand[str(name)] = demand.get(str(name), 0) + 1

    clips: list[dict[str, Any]] = []
    supply: dict[str, int] = {}
    if library_root.is_dir():
        for wav_path in sorted(library_root.rglob("*.wav")):
            relative = wav_path.relative_to(library_root).as_posix()
            sidecar = _read_json(wav_path.with_suffix(".json")) or _read_json(
                wav_path.parent / "clip.json"
            ) or {}
            event_classes = [
                str(name) for name in sidecar.get("event_classes") or []
            ]
            for name in event_classes:
                supply[name] = supply.get(name, 0) + 1
            clips.append(
                {
                    "path": relative,
                    "byte_size": wav_path.stat().st_size,
                    "event_classes": event_classes,
                    "source": sidecar.get("source"),
                    "license": sidecar.get("license"),
                    "dry": sidecar.get("dry"),
                    "notes": sidecar.get("notes"),
                    **_wav_facts(wav_path),
                }
            )

    coverage = [
        {
            "event_class": name,
            "demanding_assets": count,
            "library_clips": supply.get(name, 0),
        }
        for name, count in sorted(demand.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    unclaimed = sorted(set(supply) - set(demand))
    return {
        "schema": "avengine_studio_sound_library_v1",
        "root": str(library_root),
        "requirements": (
            "干声（无混响、无背景底噪）；混响由房间声学添加，素材自带混响会叠加。"
            "管线消费 16 kHz 单声道 PCM wav；其他采样率可以先入库，登记进正式"
            "registry 时再重采样。旁车 json 声明 event_classes（用资产档案里的"
            "词表）、source、license、dry: true"
        ),
        "coverage": coverage,
        "clips": clips,
        "clip_count": len(clips),
        "supply_without_demand": unclaimed,
    }


def sound_library_file(library_root: Path, relative: str) -> Path:
    root = library_root.resolve()
    target = (root / relative).resolve()
    if not str(target).startswith(str(root) + "/"):
        raise ProductionError("sound library path escapes the library root")
    if target.suffix not in _SOUND_FILE_SUFFIXES:
        raise ProductionError(f"not a servable sound-library file: {relative}")
    if not target.is_file():
        raise ProductionError(f"no such sound-library file: {relative}")
    return target


# --------------------------------------------------------------------------
# human verdicts


def write_human_verdict(
    task_dir: Path, *, verdict: str, note: str = "", author: str = "studio"
) -> dict[str, Any]:
    if verdict not in _VERDICT_VALUES:
        raise ProductionError(
            f"verdict must be one of {sorted(_VERDICT_VALUES)}, got {verdict!r}"
        )
    if not task_dir.is_dir():
        raise ProductionError(f"task directory does not exist: {task_dir}")
    record = {
        "schema": "avengine_studio_human_verdict_v1",
        "verdict": verdict,
        "note": str(note or "")[:2000],
        "author": str(author or "studio")[:200],
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (task_dir / _VERDICT_FILE).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def read_human_verdict(task_dir: Path) -> dict[str, Any] | None:
    return _read_json(task_dir / _VERDICT_FILE)


# --------------------------------------------------------------------------
# the stage board


def _scene_key(text: str | None) -> str | None:
    if not text:
        return None
    kujiale = re.search(r"(kujiale_\d+)", str(text))
    if kujiale:
        return kujiale.group(1)
    match = _SCENE_HASH.search(str(text))
    return match.group(1) if match else None


def _verify_room_prepare(output_dir: Path) -> dict[str, Any] | None:
    for manifest_path in sorted(output_dir.glob("*/room_manifest.json")):
        manifest = _read_json(manifest_path)
        if manifest is None:
            continue
        pairs = manifest.get("connectivity_pairs") or []
        sidecar = _read_json(manifest_path.parent / "connectivity_measurement.json")
        return {
            "scene": _scene_key(manifest.get("room_id")),
            "ok": bool(pairs),
            "summary": (
                f"连通对 {len(pairs)}"
                + (
                    f" · 测地 {sidecar['measured_geodesic_distance_m']:.1f} m"
                    if sidecar and "measured_geodesic_distance_m" in sidecar
                    else ""
                )
            ),
        }
    return None


def _worst_escape_fraction(leakage: Any) -> float | None:
    fractions: list[float] = []

    def collect(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == "escape_fraction" and isinstance(value, (int, float)):
                    fractions.append(float(value))
                else:
                    collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(leakage)
    return max(fractions) if fractions else None


def _verify_package(output_dir: Path) -> dict[str, Any] | None:
    manifest = _read_json(output_dir / "manifest.json")
    coverage = _read_json(output_dir / "semantic_material_coverage.json")
    if manifest is None and coverage is None:
        return None
    counts = (coverage or {}).get("category_triangle_counts") or {}
    total = sum(counts.values()) or 1
    sealed = bool((manifest or {}).get("package_content_sha256"))
    # The leakage probes are the one QA a frame bug cannot survive: probes
    # planted in room coordinates escape a mis-framed mesh wholesale. The
    # sideways incident sat at 0.87-0.94 while the board painted this cell
    # green from the coverage report alone. Never again: a package whose
    # probes mostly miss its own geometry is red no matter what else passed.
    worst_escape = _worst_escape_fraction(
        _read_json(output_dir / "qa" / "ray_leakage.json") or {}
    )
    frame_suspect = worst_escape is not None and worst_escape > 0.5
    escape_text = (
        "" if worst_escape is None else f" · 探针逃逸最坏 {100 * worst_escape:.0f}%"
    )
    return {
        "scene": _scene_key((coverage or {}).get("room_id")),
        "ok": sealed and coverage is not None and not frame_suspect,
        "summary": (
            (
                f"{(coverage or {}).get('surface_count', '?')} 类 · 默认材质 "
                f"{(coverage or {}).get('unknown_semantic_category_count', '?')} 类 · "
                f"未标注 {100.0 * counts.get('unannotated', 0) / total:.1f}%"
                + escape_text
                + ("（疑似坐标系错误：探针大量在几何体外）" if frame_suspect else "")
            )
            if coverage
            else "封印在但覆盖率报告缺失"
        ),
    }


def _verify_route_bank(output_dir: Path) -> dict[str, Any] | None:
    report = _read_json(output_dir / "route_report.json")
    if report is None:
        return None
    scenes = report.get("scenes") or []
    first = scenes[0] if scenes and isinstance(scenes[0], Mapping) else {}
    return {
        "scene": _scene_key(str(first.get("scene", ""))),
        "ok": int(report.get("floors_with_routes", 0)) > 0,
        "summary": (
            f"楼层 {report.get('floors_with_routes', 0)}/"
            f"{report.get('floors_examined', 0)} 有路线"
        ),
    }


def _verify_episode(output_dir: Path) -> dict[str, Any] | None:
    receipt = _read_json(output_dir / "receipt.json")
    if receipt is None:
        return None
    # The machine audition is the acceptance document when it exists: it
    # already judged direction, channels, levels, frames, mux and the pose
    # identity chain, and wrote its reasons. Episodes rendered before the
    # audition existed fall back to the raw direction numbers, labelled so.
    audition = _read_json(output_dir / "machine_audition.json")
    if audition is not None:
        passed = audition.get("verdict") == "pass"
        return {
            "scene": _scene_key(receipt.get("scene_id")),
            "ok": passed and (output_dir / "episode_binaural.mp4").is_file(),
            "summary": f"机器听审{'过' if passed else '不过'} · "
            + str(audition.get("summary_zh") or "")[:160],
        }
    report = _read_json(output_dir / "audio_foa" / "render_report.json") or {}
    rendered = report.get("frames_rendered")
    within = report.get("frames_within_tolerance")
    error = report.get("direction_error_deg") or {}
    return {
        "scene": _scene_key(receipt.get("scene_id")),
        "ok": (output_dir / "episode_binaural.mp4").is_file(),
        "summary": (
            f"帧 {within}/{rendered} 达容差 · 误差中位 "
            f"{error.get('median', '?')}°（旧任务：无机器听审档）"
            if rendered is not None
            else "回执在但逐帧报告缺失"
        ),
    }


def _verify_kujiale_routes(output_dir: Path) -> dict[str, Any] | None:
    """The kujiale bank speaks its own artifact dialect, not route_report.json.

    Its acceptance evidence is the pair of gate documents the tool itself
    refuses to pass without: trajectory_coverage.json and
    trajectory_diversity.json, plus the bank whose episode_count they judged.
    """

    bank = _read_json(output_dir / "trajectory_bank.json")
    if bank is None:
        return None
    coverage = _read_json(output_dir / "trajectory_coverage.json") or {}
    diversity = _read_json(output_dir / "trajectory_diversity.json") or {}
    delivery = _read_json(output_dir / "delivery.json") or {}
    # The coverage record carries the gate thresholds beside the observed
    # numbers rather than a verdict string, so the verdict is recomputed here
    # from the same pair the tool gated on.
    gate = coverage.get("gate") or {}
    try:
        coverage_ok = float(
            coverage.get("coverage_fraction_by_threshold", {}).get("0.50")
            or gate.get("observed_fraction_within_0.50m", 0.0)
        ) >= float(gate.get("minimum_fraction_within_0.50m", 1.0)) and float(
            gate.get("observed_maximum_gap_m", float("inf"))
        ) <= float(gate.get("maximum_gap_m", 0.0))
    except (TypeError, ValueError):
        coverage_ok = False
    diversity_ok = str(diversity.get("status") or "") == "pass"
    return {
        "scene": _scene_key(delivery.get("room_id") or bank.get("schema")),
        "ok": bool(bank.get("episode_count")) and coverage_ok and diversity_ok,
        "summary": (
            f"{bank.get('episode_count', 0)} 条轨迹 · 覆盖 "
            f"{'过' if coverage_ok else '未过'} · 槽位多样性 "
            f"{'过' if diversity_ok else '未过'}"
        ),
    }


def _verify_kujiale_package(output_dir: Path) -> dict[str, Any] | None:
    receipt = _read_json(output_dir / "receipt.json")
    if receipt is None:
        return None
    rlr_manifest = receipt.get("rlr_manifest")
    loadable = bool(rlr_manifest) and Path(str(rlr_manifest)).is_file()
    return {
        "scene": _scene_key(receipt.get("room_id")),
        "ok": loadable,
        "summary": (
            f"RLR 可加载 · 去退化面 {receipt.get('removed_degenerate_triangles')}"
            f" · {receipt.get('derived_triangle_count')} 三角面"
            if loadable
            else "回执在但 RLR 包缺失"
        ),
    }


_STAGE_VERIFIERS = {
    "hm3d_room_prepare": ("room", _verify_room_prepare),
    "semantic_acoustic_package": ("package", _verify_package),
    "hm3d_route_bank": ("routes", _verify_route_bank),
    "hm3d_episode": ("episode", _verify_episode),
    "kujiale_route_bank": ("routes", _verify_kujiale_routes),
    "kujiale_acoustic_package": ("package", _verify_kujiale_package),
}

BOARD_STAGES = ("room", "package", "routes", "episode", "verdict")


def board_rows(task_records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """One row per scene, one cell per stage, painted from verifier artifacts.

    Newest task wins a cell: a re-run that fixed a scene should replace the
    failure it fixed, and the loser is still reachable through the task list.
    """

    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for record in task_records:
        template = str(record.get("template") or "")
        entry = _STAGE_VERIFIERS.get(template)
        if entry is None:
            continue
        stage, verifier = entry
        output_dir = Path(str(record.get("output_dir") or ""))
        verified = verifier(output_dir) if output_dir.is_dir() else None
        if verified is None:
            # A task still running has not failed its verification - it has
            # not reached it. Say which, or a slow stage reads as a broken one.
            still_working = str(record.get("status")) in ("queued", "running")
            verified = {
                "scene": None,
                "ok": False,
                "summary": "任务尚在运行" if still_working else "无验收产物",
            }
        scene = verified.pop("scene", None) or (
            "kujiale_0020" if template.startswith("kujiale") else "unknown"
        )
        cell = {
            "task_id": record.get("task_id"),
            "created_at": record.get("created_at"),
            "task_status": record.get("status"),
            **verified,
        }
        row = cells.setdefault(scene, {})
        current = row.get(stage)
        if current is None or str(cell.get("created_at") or "") > str(
            current.get("created_at") or ""
        ):
            row[stage] = cell

        if template == "hm3d_episode":
            # The verdict column is machine-first: the audition the episode
            # chain writes is the acceptance, and a human verdict - when one
            # was recorded at all - overrides it for the same task. Ordering
            # across tasks follows the task's own created_at, like every
            # other cell, so a re-run replaces the episode it replaced.
            verdict_cell = None
            audition = _read_json(output_dir / "machine_audition.json")
            if audition is not None:
                verdict_cell = {
                    "task_id": record.get("task_id"),
                    "created_at": record.get("created_at"),
                    "task_status": record.get("status"),
                    "ok": audition.get("verdict") == "pass",
                    "summary": "机器 "
                    + str(audition.get("verdict"))
                    + " · "
                    + str(audition.get("summary_zh") or "")[:120],
                }
            verdict = read_human_verdict(Path(str(record.get("task_dir") or "")))
            if verdict is not None:
                verdict_cell = {
                    "task_id": record.get("task_id"),
                    "created_at": record.get("created_at"),
                    "task_status": record.get("status"),
                    "ok": verdict.get("verdict") == "pass",
                    "summary": (
                        f"人工覆核 {verdict.get('verdict')}"
                        + (f" · {verdict.get('note')}" if verdict.get("note") else "")
                    ),
                }
            if verdict_cell is not None:
                current_verdict = row.get("verdict")
                if current_verdict is None or str(
                    verdict_cell.get("created_at") or ""
                ) > str(current_verdict.get("created_at") or ""):
                    row["verdict"] = verdict_cell

    rows = [
        {"scene": scene, "cells": stage_cells}
        for scene, stage_cells in sorted(cells.items())
    ]
    return {
        "schema": "avengine_studio_board_v1",
        "stages": list(BOARD_STAGES),
        "note": (
            "格子的颜色来自各工序自己的验收产物，不来自任务退出码；"
            "同一格多次运行以最新为准，旧任务仍在任务列表里"
        ),
        "rows": rows,
    }


# --------------------------------------------------------------------------
# the review queue


def review_queue(task_records: list[Mapping[str, Any]]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for record in task_records:
        if str(record.get("template") or "") != "hm3d_episode":
            continue
        output_dir = Path(str(record.get("output_dir") or ""))
        receipt = _read_json(output_dir / "receipt.json")
        if receipt is None:
            continue
        report = _read_json(output_dir / "audio_foa" / "render_report.json") or {}
        per_frame = report.get("per_frame") or []
        artifacts = {
            "mp4": "episode_binaural.mp4",
            "foa_stereo_fold": "audio_foa/moving_source.stereo_fold.wav",
            "first_frame": "video/frame_0000.png",
            "foa_report": "audio_foa/render_report.json",
            "machine_audition": "machine_audition.json",
            "receipt": "receipt.json",
        }
        available = {
            name: relative
            for name, relative in artifacts.items()
            if (output_dir / relative).is_file()
        }
        audition = _read_json(output_dir / "machine_audition.json")
        episodes.append(
            {
                "task_id": record.get("task_id"),
                "created_at": record.get("created_at"),
                "machine_audition": (
                    {
                        "verdict": audition.get("verdict"),
                        "summary_zh": audition.get("summary_zh"),
                        "scope_note_zh": audition.get("scope_note_zh"),
                        "checks": [
                            {
                                "name": check.get("name"),
                                "status": check.get("status"),
                                "reason_zh": check.get("reason_zh"),
                            }
                            for check in audition.get("checks") or []
                            if isinstance(check, Mapping)
                        ],
                    }
                    if audition is not None
                    else None
                ),
                "scene_id": receipt.get("scene_id"),
                "motion_case": receipt.get("motion_case"),
                "episode_index": receipt.get("episode_index"),
                "episode_id": receipt.get("episode_id"),
                "frames_rendered": report.get("frames_rendered"),
                "frames_within_tolerance": report.get("frames_within_tolerance"),
                "direction_error_deg": report.get("direction_error_deg"),
                "reverberation": report.get("reverberation"),
                "error_deg_per_frame": [
                    frame.get("error_deg")
                    for frame in per_frame
                    if isinstance(frame, Mapping)
                ],
                "artifacts": available,
                "verdict": read_human_verdict(
                    Path(str(record.get("task_dir") or ""))
                ),
            }
        )
    episodes.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {"schema": "avengine_studio_review_queue_v1", "episodes": episodes}
