#!/usr/bin/env python3
"""Join F2 direction facts with native windowed pixel/audio evidence.

This tool binds one main fact and its Gate A fact to the same native pixel
truth.  It checks the fact-declared query visibility state at every frame in
the declared window, then records the independent visual and audio verifier
results.  The output remains a research candidate and never performs dataset
admission.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "avengine_qa_v3_f2_direction_pixel_join_v1"
VISIBLE_STATES = frozenset({"visible_clear", "visible_occluded"})
VISIBILITY_STATES = frozenset({"any", "visible", "out_of_view"})


class F2DirectionPixelJoinError(ValueError):
    """An input is not a readable instance of the declared evidence format."""


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise F2DirectionPixelJoinError(f"cannot read JSON {path}: {exc}") from exc


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise F2DirectionPixelJoinError(f"{owner} must be an object")
    return value


def _resolve_input(path: str | Path, *, owner: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise F2DirectionPixelJoinError(f"{owner} is missing: {value}")
    return value


def _resolve_directory(path: str | Path, *, owner: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_dir():
        raise F2DirectionPixelJoinError(f"{owner} is missing: {value}")
    return value


def _resolve_declared_path(
    raw: Any, *, base: Path, owner: str, directory: bool = False,
) -> Path:
    if isinstance(raw, Mapping):
        raw = raw.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise F2DirectionPixelJoinError(f"{owner} must be a non-empty path")
    value = Path(raw.strip()).expanduser()
    if not value.is_absolute():
        value = base / value
    return (
        _resolve_directory(value, owner=owner)
        if directory
        else _resolve_input(value, owner=owner)
    )


def _value_from_containers(
    fact: Mapping[str, Any], key: str,
) -> Any:
    for container in (
        fact,
        fact.get("truth"),
        fact.get("open"),
        fact.get("geometry"),
    ):
        if isinstance(container, Mapping) and key in container:
            return container[key]
    return None


def _required_string(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise F2DirectionPixelJoinError(f"{owner} must be a non-empty string")
    return value.strip()


def _fact_artifact_path(
    fact: Mapping[str, Any],
    fact_path: Path,
    *,
    keys: tuple[str, ...],
    default_name: str,
    owner: str,
) -> Path:
    raw = None
    artifacts = fact.get("artifacts")
    if isinstance(artifacts, Mapping):
        for key in keys:
            if artifacts.get(key) is not None:
                raw = artifacts[key]
                break
    if raw is None:
        for key in keys:
            if fact.get(key) is not None:
                raw = fact[key]
                break
    if raw is None:
        raw = default_name
    return _resolve_declared_path(
        raw, base=fact_path.parent, owner=owner
    )


def _window(fact: Mapping[str, Any], *, owner: str) -> tuple[int, int]:
    value = _value_from_containers(fact, "query_window_frame_bounds")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] < 0
        or value[1] < value[0]
    ):
        raise F2DirectionPixelJoinError(
            f"{owner}.query_window_frame_bounds must be [lo, hi]"
        )
    return int(value[0]), int(value[1])


def _query_visibility(fact: Mapping[str, Any], *, owner: str) -> str:
    value = _value_from_containers(fact, "query_visibility")
    if value is not None:
        if not isinstance(value, str) or value.strip().lower() not in VISIBILITY_STATES:
            raise F2DirectionPixelJoinError(
                f"{owner}.query_visibility must be one of "
                f"{sorted(VISIBILITY_STATES)}"
            )
        return value.strip().lower()

    # Compatibility for facts emitted before the tri-state declaration.
    legacy = _value_from_containers(fact, "query_requires_visibility")
    if legacy is not None:
        if not isinstance(legacy, bool):
            raise F2DirectionPixelJoinError(
                f"{owner}.query_requires_visibility must be a boolean"
            )
        return "visible" if legacy else "any"
    return "any"


def _geometry_status(value: Any, *, role: str) -> str | None:
    if not isinstance(value, Mapping):
        return None
    role_value = value.get(role)
    if isinstance(role_value, Mapping):
        status = role_value.get("status")
        if isinstance(status, str):
            return status
        passed = role_value.get("passes")
        if isinstance(passed, bool):
            return "pass" if passed else "fail"
        return None
    if isinstance(role_value, str):
        return role_value
    status = value.get("status")
    if isinstance(status, str):
        return status
    passed = value.get("passes")
    return ("pass" if passed else "fail") if isinstance(passed, bool) else None


def _window_geometry_detail(
    main_fact: Mapping[str, Any],
    gatea_fact: Mapping[str, Any],
    *,
    role: str,
) -> Mapping[str, Any] | None:
    for fact in (main_fact, gatea_fact):
        for container in (
            fact,
            fact.get("generation_checks"),
            fact.get("truth"),
            fact.get("open"),
        ):
            if not isinstance(container, Mapping):
                continue
            value = container.get("query_visibility_window_geometry")
            if not isinstance(value, Mapping):
                continue
            role_value = value.get(role)
            if isinstance(role_value, Mapping):
                return role_value
            if role_value is not None:
                return None
            # A direct status is meaningful only on the matching fact. Do
            # not copy one aggregate/direct status to both main and GateA.
            if fact.get("variant") == role and (
                "status" in value or "passes" in value
            ):
                return value
    return None


def _window_geometry_status(
    main_fact: Mapping[str, Any],
    gatea_fact: Mapping[str, Any],
    *,
    role: str,
) -> str | None:
    detail = _window_geometry_detail(main_fact, gatea_fact, role=role)
    return _geometry_status(detail, role=role) if detail is not None else None


def _pixel_truth(raw: Mapping[str, Any], *, path: Path) -> Mapping[str, Any]:
    if raw.get("status") is not None and raw.get("status") != "pass":
        raise F2DirectionPixelJoinError(
            f"pixel evidence {path} must have status='pass'"
        )
    inline = raw.get("pixel_visibility")
    if isinstance(inline, Mapping):
        return inline

    artifacts = raw.get("artifacts")
    if isinstance(artifacts, Mapping):
        truth_raw = artifacts.get("truth")
        if isinstance(truth_raw, str) and truth_raw.strip():
            truth_path = Path(truth_raw).expanduser()
            if not truth_path.is_absolute():
                truth_path = path.parent / truth_path
            truth = _mapping(_read(truth_path.resolve()), owner="pixel truth")
            nested = truth.get("pixel_visibility")
            return _mapping(
                nested if isinstance(nested, Mapping) else truth,
                owner="pixel truth",
            )
    return raw


def _frames_by_slot(
    truth: Mapping[str, Any],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    instances = _mapping(truth.get("per_instance"), owner="pixel_truth.per_instance")
    result: dict[str, dict[int, Mapping[str, Any]]] = {}
    for raw_slot, raw_record in instances.items():
        slot = _required_string(raw_slot, owner="pixel_truth.per_instance key")
        record = _mapping(
            raw_record, owner=f"pixel_truth.per_instance.{slot}"
        )
        frames = record.get("frames")
        if not isinstance(frames, list):
            raise F2DirectionPixelJoinError(
                f"pixel_truth.per_instance.{slot}.frames must be a list"
            )
        by_frame: dict[int, Mapping[str, Any]] = {}
        for index, raw_frame in enumerate(frames):
            frame = _mapping(
                raw_frame,
                owner=f"pixel_truth.per_instance.{slot}.frames[{index}]",
            )
            value = frame.get("frame_index")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise F2DirectionPixelJoinError(
                    f"pixel_truth.per_instance.{slot}.frames[{index}] "
                    "has an invalid frame_index"
                )
            if value in by_frame:
                raise F2DirectionPixelJoinError(
                    f"pixel_truth.per_instance.{slot} repeats frame {value}"
                )
            by_frame[int(value)] = frame
        result[slot] = by_frame
    return result


def _evaluate_fact(
    fact: Mapping[str, Any],
    *,
    label: str,
    frames_by_slot: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    slot = _required_string(
        fact.get("target_slot"), owner=f"{label}.target_slot"
    )
    lo, hi = _window(fact, owner=label)
    state = _query_visibility(fact, owner=label)
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    slot_frames = frames_by_slot.get(slot)
    if slot_frames is None:
        return (
            {
                "target_slot": slot,
                "query_visibility": state,
                "query_window_frame_bounds": [lo, hi],
                "passed": False,
                "frames": [],
            },
            [f"{label}.{slot}.missing_from_pixel_truth"],
        )

    for frame_index in range(lo, hi + 1):
        frame = slot_frames.get(frame_index)
        if frame is None:
            reasons.append(f"{label}.{slot}.frame_{frame_index}.missing")
            rows.append(
                {
                    "frame_index": frame_index,
                    "state": None,
                    "passed": False,
                }
            )
            continue
        actual = frame.get("state")
        if state == "out_of_view":
            passed = actual == "out_of_view"
        elif state == "visible":
            passed = actual in VISIBLE_STATES
        else:
            passed = True
        if not passed:
            reasons.append(
                f"{label}.{slot}.frame_{frame_index}."
                f"expected_{state}_got_{actual}"
            )
        rows.append(
            {
                "frame_index": frame_index,
                "state": actual,
                "passed": passed,
            }
        )
    return (
        {
            "target_slot": slot,
            "query_visibility": state,
            "query_window_frame_bounds": [lo, hi],
            "passed": not reasons,
            "frames": rows,
        },
        reasons,
    )


def _visual_binding_checks(
    visual: Mapping[str, Any],
    *,
    visual_path: Path,
    point_id: str,
) -> tuple[dict[str, Any], list[str]]:
    inputs = _mapping(visual.get("inputs"), owner="visual verification.inputs")
    selection_path = _resolve_declared_path(
        inputs.get("selection_manifest"),
        base=visual_path.parent,
        owner="visual verification.inputs.selection_manifest",
    )
    visual_root = _resolve_declared_path(
        inputs.get("visual_root"),
        base=visual_path.parent,
        owner="visual verification.inputs.visual_root",
        directory=True,
    )
    selection = _mapping(
        _read(selection_path), owner="visual selection manifest"
    )
    selected = selection.get("selected")
    selected_ids = {
        item.get("point_id")
        for item in selected
        if isinstance(item, Mapping) and isinstance(item.get("point_id"), str)
    } if isinstance(selected, list) else set()
    points = visual.get("points")
    point_rows = [
        item for item in points
        if isinstance(item, Mapping) and item.get("point_id") == point_id
    ] if isinstance(points, list) else []
    point_dir = visual_root / point_id
    checks = {
        "selection_manifest": str(selection_path),
        "selected_point": point_id in selected_ids,
        "visual_root": str(visual_root),
        "point_directory": str(point_dir),
        "point_directory_exists": point_dir.is_dir(),
        "point_rows": len(point_rows),
        "point_status_pass": (
            len(point_rows) == 1 and point_rows[0].get("status") == "pass"
        ),
    }
    reasons: list[str] = []
    if point_id not in selected_ids:
        reasons.append("visual_selection_manifest_missing_point")
    if not point_dir.is_dir():
        reasons.append("visual_root_missing_point")
    if len(point_rows) != 1 or point_rows[0].get("status") != "pass":
        reasons.append("visual_report_missing_passing_point")
    return checks, reasons


def _fact_program_id(fact: Mapping[str, Any], *, owner: str) -> str:
    audio = _mapping(fact.get("audio"), owner=f"{owner}.audio")
    return _required_string(audio.get("program_id"), owner=f"{owner}.audio.program_id")


def _audio_binding_checks(
    audio: Mapping[str, Any],
    *,
    audio_path: Path,
    main_fact: Mapping[str, Any],
    gatea_fact: Mapping[str, Any],
    main_fact_path: Path,
    gatea_fact_path: Path,
    point_id: str,
) -> tuple[dict[str, Any], list[str]]:
    root = _resolve_declared_path(
        audio.get("audio_root"),
        base=audio_path.parent,
        owner="audio verification.audio_root",
        directory=True,
    )
    expected_names = {
        "main": point_id,
        "gateA": f"{point_id}_gateA",
    }
    checked = audio.get("checked_renders")
    waveform_pairs = audio.get("audio_variant_waveform_nonidentity_pairs")
    semantic_pairs = audio.get("gatea_semantic_flip_pairs")
    reasons: list[str] = []
    scalar_checks = {
        "audio_root": str(root),
        "checked_renders": checked,
        "waveform_nonidentity_pairs": waveform_pairs,
        "semantic_flip_pairs": semantic_pairs,
        "checked_renders_sufficient": (
            isinstance(checked, int) and not isinstance(checked, bool) and checked >= 2
        ),
        "waveform_nonidentity_sufficient": (
            isinstance(waveform_pairs, int)
            and not isinstance(waveform_pairs, bool)
            and waveform_pairs >= 1
        ),
        "semantic_flip_sufficient": (
            isinstance(semantic_pairs, int)
            and not isinstance(semantic_pairs, bool)
            and semantic_pairs >= 1
        ),
    }
    if not scalar_checks["checked_renders_sufficient"]:
        reasons.append("audio_checked_renders_insufficient")
    if not scalar_checks["waveform_nonidentity_sufficient"]:
        reasons.append("audio_waveform_nonidentity_not_established")
    if not scalar_checks["semantic_flip_sufficient"]:
        reasons.append("audio_semantic_flip_not_established")

    execution = _mapping(
        audio.get("execution_variant_verification"),
        owner="audio verification.execution_variant_verification",
    )
    verified_renders = execution.get("verified_renders")
    execution_verified = (
        execution.get("status") == "verified"
        and isinstance(verified_renders, list)
        and all(name in verified_renders for name in expected_names.values())
    )
    scalar_checks["execution_variant_verified"] = execution_verified
    scalar_checks["verified_renders"] = verified_renders
    if not execution_verified:
        reasons.append("audio_execution_variants_not_verified_for_pair")

    fact_by_variant = {"main": main_fact, "gateA": gatea_fact}
    fact_path_by_variant = {"main": main_fact_path, "gateA": gatea_fact_path}
    render_checks: dict[str, Any] = {}
    program_variant_ids: dict[str, Any] = {}
    for variant, name in expected_names.items():
        render_dir = root / name
        receipt_path = render_dir / "research_receipt.json"
        row: dict[str, Any] = {
            "render_name": name,
            "directory": str(render_dir),
            "directory_exists": render_dir.is_dir(),
            "receipt": str(receipt_path),
            "receipt_exists": receipt_path.is_file(),
        }
        if not render_dir.is_dir() or not receipt_path.is_file():
            reasons.append(f"audio_{variant}_render_missing")
            render_checks[variant] = row
            continue
        receipt = _mapping(_read(receipt_path), owner=f"{variant} audio receipt")
        program = _mapping(
            receipt.get("audio_program"),
            owner=f"{variant} audio receipt.audio_program",
        )
        fact = fact_by_variant[variant]
        expected_program_id = _fact_program_id(
            fact, owner=f"{variant} fact"
        )
        row.update({
            "receipt_status": receipt.get("status"),
            "execution_variant": receipt.get("execution_variant"),
            "program_id": program.get("program_id"),
            "program_variant_id": program.get("variant_id"),
            "program_path": program.get("path"),
        })
        program_variant_ids[variant] = program.get("variant_id")
        if receipt.get("status") != "pass":
            reasons.append(f"audio_{variant}_receipt_not_pass")
        if receipt.get("execution_variant") != variant:
            reasons.append(f"audio_{variant}_execution_variant_mismatch")
        if program.get("program_id") != expected_program_id:
            reasons.append(f"audio_{variant}_program_id_mismatch")
        program_path = program.get("path")
        if (
            not isinstance(program_path, str)
            or not Path(program_path).expanduser().is_file()
        ):
            reasons.append(f"audio_{variant}_program_path_missing")
        inputs = _mapping(
            receipt.get("inputs"), owner=f"{variant} audio receipt.inputs"
        )
        expected_m1 = _fact_artifact_path(
            fact,
            fact_path_by_variant[variant],
            keys=("m1_capture_request", "m1_request"),
            default_name="m1_capture_request.json",
            owner=f"{variant} fact M1 request",
        )
        expected_endpoint = _fact_artifact_path(
            fact,
            fact_path_by_variant[variant],
            keys=("source_endpoint_registry", "source_endpoints"),
            default_name="source_endpoints.json",
            owner=f"{variant} fact endpoint registry",
        )
        for field, expected in (
            ("m1_request", expected_m1),
            ("source_endpoint_registry", expected_endpoint),
        ):
            actual = _resolve_declared_path(
                inputs.get(field),
                base=receipt_path.parent,
                owner=f"{variant} audio receipt.inputs.{field}",
            )
            same = actual == expected
            row[f"{field}_path_equal"] = same
            if not same:
                reasons.append(f"audio_{variant}_{field}_path_mismatch")
        row["mixture_exists"] = (
            (render_dir / "audio" / "binaural" / "mixture.wav").is_file()
        )
        if not row["mixture_exists"]:
            reasons.append(f"audio_{variant}_mixture_missing")
        render_checks[variant] = row
    if (
        "main" in program_variant_ids
        and "gateA" in program_variant_ids
        and (
            not isinstance(program_variant_ids["main"], str)
            or not isinstance(program_variant_ids["gateA"], str)
            or program_variant_ids["main"] != program_variant_ids["gateA"]
        )
    ):
        reasons.append("audio_program_variant_id_changed_between_main_and_gateA")
    return {
        **scalar_checks,
        "render_checks": render_checks,
        "program_variant_ids": program_variant_ids,
    }, reasons


def _verification_checks(
    visual: Mapping[str, Any],
    audio: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    visual_pass = visual.get("status") == "pass"
    if not visual_pass:
        reasons.append(f"visual_verification_status_{visual.get('status')!r}")

    failures = audio.get("failures")
    audio_failures_empty = isinstance(failures, list) and not failures
    if not audio_failures_empty:
        reasons.append("audio_verification_failures_not_empty")

    execution = audio.get("execution_variant_verification")
    if isinstance(execution, Mapping):
        execution_status = execution.get("status")
    else:
        execution_status = execution
    execution_verified = execution_status == "verified"
    if not execution_verified:
        reasons.append(
            f"execution_variant_verification_status_{execution_status!r}"
        )
    return (
        {
            "visual_status": visual.get("status"),
            "visual_pass": visual_pass,
            "audio_failures": failures,
            "audio_failures_empty": audio_failures_empty,
            "execution_variant_verification": execution,
            "execution_variant_verified": execution_verified,
        },
        reasons,
    )


def join(
    main_fact_path: str | Path,
    gatea_fact_path: str | Path,
    visual_verification_path: str | Path,
    audio_verification_path: str | Path,
    pixel_evidence_path: str | Path,
) -> dict[str, Any]:
    main_path = _resolve_input(main_fact_path, owner="main fact")
    gatea_path = _resolve_input(gatea_fact_path, owner="GateA fact")
    visual_path = _resolve_input(
        visual_verification_path, owner="visual verification"
    )
    audio_path = _resolve_input(
        audio_verification_path, owner="audio verification"
    )
    pixel_path = _resolve_input(pixel_evidence_path, owner="pixel evidence")

    main = _mapping(_read(main_path), owner="main fact")
    gatea = _mapping(_read(gatea_path), owner="GateA fact")
    point_id = _required_string(main.get("point_id"), owner="main fact.point_id")
    scene_id = _required_string(main.get("scene_id"), owner="main fact.scene_id")
    profile_id = _required_string(
        main.get("profile_id"), owner="main fact.profile_id"
    )
    reasons: list[str] = []
    if main.get("variant") != "main":
        reasons.append("main_fact_variant_must_be_main")
    if gatea.get("variant") != "gateA":
        reasons.append("gateA_fact_variant_must_be_gateA")
    for label, fact in (("main", main), ("GateA", gatea)):
        if fact.get("point_id") != point_id:
            raise F2DirectionPixelJoinError(
                f"{label} fact point_id differs from main fact"
            )
        if fact.get("scene_id") != scene_id:
            raise F2DirectionPixelJoinError(
                f"{label} fact scene_id differs from main fact"
            )
        if fact.get("profile_id") != profile_id:
            raise F2DirectionPixelJoinError(
                f"{label} fact profile_id differs from main fact"
            )
    if gatea.get("gatea_of") is not None and gatea.get("gatea_of") != point_id:
        reasons.append("gateA_fact_gatea_of_mismatch")

    main_target = _required_string(
        main.get("target_slot"), owner="main fact.target_slot"
    )
    gatea_target = _required_string(
        gatea.get("target_slot"), owner="GateA fact.target_slot"
    )
    if main_target == gatea_target:
        reasons.append("main_gateA_target_slot_not_exchanged")

    main_window = _window(main, owner="main fact")
    gatea_window = _window(gatea, owner="GateA fact")
    if main_window != gatea_window:
        reasons.append("main_gatea_query_windows_differ")

    geometry_details = {
        "main": _window_geometry_detail(main, gatea, role="main"),
        "gateA": _window_geometry_detail(main, gatea, role="gateA"),
    }
    geometry_checks = {
        "main_status": (
            _geometry_status(geometry_details["main"], role="main")
            if geometry_details["main"] is not None else None
        ),
        "gateA_status": (
            _geometry_status(geometry_details["gateA"], role="gateA")
            if geometry_details["gateA"] is not None else None
        ),
    }
    geometry_checks.update({
        "main_pass": geometry_checks["main_status"] == "pass",
        "gateA_pass": geometry_checks["gateA_status"] == "pass",
    })
    geometry_checks["both_pass"] = (
        geometry_checks["main_pass"] and geometry_checks["gateA_pass"]
    )
    for label in ("main", "gateA"):
        detail = geometry_details[label]
        if not geometry_checks[f"{label}_pass"]:
            reasons.append(f"{label}_query_visibility_window_geometry_not_pass")
            continue
        fact = main if label == "main" else gatea
        expected_window = main_window if label == "main" else gatea_window
        expected_policy = _query_visibility(fact, owner=f"{label} fact")
        if detail.get("frame_bounds") != list(expected_window):
            reasons.append(
                f"{label}_query_visibility_window_geometry_window_mismatch"
            )
        if detail.get("policy") != expected_policy:
            reasons.append(
                f"{label}_query_visibility_window_geometry_policy_mismatch"
            )
        if detail.get("source_slot_id") != fact.get("target_slot"):
            reasons.append(
                f"{label}_query_visibility_window_geometry_slot_mismatch"
            )

    visual = _mapping(_read(visual_path), owner="visual verification")
    audio = _mapping(_read(audio_path), owner="audio verification")
    visual_binding, visual_binding_reasons = _visual_binding_checks(
        visual, visual_path=visual_path, point_id=point_id
    )
    reasons.extend(visual_binding_reasons)
    verifier_checks, verifier_reasons = _verification_checks(visual, audio)
    reasons.extend(verifier_reasons)
    audio_binding, audio_binding_reasons = _audio_binding_checks(
        audio,
        audio_path=audio_path,
        main_fact=main,
        gatea_fact=gatea,
        main_fact_path=main_path,
        gatea_fact_path=gatea_path,
        point_id=point_id,
    )
    reasons.extend(audio_binding_reasons)

    pixel_wrapper = _mapping(_read(pixel_path), owner="pixel evidence")
    if pixel_wrapper.get("schema") != "qa_v3_current_timeline_native_pixel_probe_v1":
        raise F2DirectionPixelJoinError(
            "pixel evidence must have schema "
            "'qa_v3_current_timeline_native_pixel_probe_v1'"
        )
    if pixel_wrapper.get("status") != "pass":
        raise F2DirectionPixelJoinError("pixel evidence must have status='pass'")
    pixel_inputs = _mapping(
        pixel_wrapper.get("inputs"), owner="pixel evidence.inputs"
    )
    expected_selection = _fact_artifact_path(
        main,
        main_path,
        keys=("actor_selection", "selection"),
        default_name="actor_selection.json",
        owner="main fact actor selection",
    )
    expected_timeline = _fact_artifact_path(
        main,
        main_path,
        keys=("timeline",),
        default_name="timeline.json",
        owner="main fact timeline",
    )
    pixel_path_checks: dict[str, Any] = {}
    for field, expected in (
        ("actor_selection", expected_selection),
        ("timeline", expected_timeline),
    ):
        actual = _resolve_declared_path(
            pixel_inputs.get(field),
            base=pixel_path.parent,
            owner=f"pixel evidence.inputs.{field}",
        )
        same = actual == expected
        pixel_path_checks[f"{field}_path_equal"] = same
        if not same:
            reasons.append(f"pixel_{field}_path_mismatch")
    expected_map = (
        (main.get("room") or {}).get("native_map")
        if isinstance(main.get("room"), Mapping)
        else None
    )
    native_map = _required_string(
        pixel_wrapper.get("native_map"), owner="pixel evidence.native_map"
    )
    if not isinstance(expected_map, str) or not expected_map:
        raise F2DirectionPixelJoinError(
            "main fact.room.native_map is required to bind pixel evidence"
        )
    pixel_path_checks["native_map"] = native_map
    pixel_path_checks["native_map_matches_fact"] = native_map == expected_map
    if native_map != expected_map:
        reasons.append("pixel_native_map_mismatch")

    truth = _pixel_truth(pixel_wrapper, path=pixel_path)
    if truth.get("schema") != "avengine_qa_pixel_visibility_truth_v1":
        raise F2DirectionPixelJoinError(
            "pixel truth must have schema 'avengine_qa_pixel_visibility_truth_v1'"
        )
    if truth.get("status") != "computed_modal_target_only_v1":
        raise F2DirectionPixelJoinError(
            "pixel truth must have status='computed_modal_target_only_v1'"
        )
    frames_by_slot = _frames_by_slot(truth)
    main_eval, main_reasons = _evaluate_fact(
        main, label="main", frames_by_slot=frames_by_slot
    )
    gatea_eval, gatea_reasons = _evaluate_fact(
        gatea, label="GateA", frames_by_slot=frames_by_slot
    )
    reasons.extend(main_reasons)
    reasons.extend(gatea_reasons)

    passed = not reasons
    report = {
        "schema": SCHEMA,
        "status": "research_candidate",
        "pixel_join_status": "pass" if passed else "pixel_rejected",
        "evidence_class": (
            "pixel_qualified_candidate" if passed else "pixel_rejected"
        ),
        "qualification_claim": False,
        "point_id": point_id,
        "scene_id": scene_id,
        "profile_id": profile_id,
        "rejection_reasons": reasons,
        "checks": {
            "fact_variants": {
                "main": main.get("variant"),
                "gateA": gatea.get("variant"),
                "main_gateA_target_slots_exchanged": (
                    main_target != gatea_target
                ),
            },
            "query_visibility_window_geometry": geometry_checks,
            "visual_binding": visual_binding,
            "visual_audio_verification": verifier_checks,
            "audio_binding": audio_binding,
            "pixel_binding": pixel_path_checks,
            "main_window": main_eval,
            "gateA_window": gatea_eval,
            "pixel_truth_schema": truth.get("schema"),
            "pixel_truth_status": truth.get("status"),
            "pixel_truth_frame_indices": truth.get("frame_indices"),
        },
        "inputs": {
            "main_fact": str(main_path),
            "gateA_fact": str(gatea_path),
            "visual_verification": str(visual_path),
            "audio_verification": str(audio_path),
            "pixel_evidence": str(pixel_path),
        },
        "boundary": (
            "Native pixel visibility joined to declared main/Gate A direction "
            "windows. The result remains a research candidate and does not "
            "establish dataset admission or human answerability."
        ),
    }
    return report


def _write_fresh(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise F2DirectionPixelJoinError(
            f"refusing to overwrite output: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-fact", "--fact", dest="main_fact", required=True, type=Path)
    parser.add_argument("--gatea-fact", required=True, type=Path)
    parser.add_argument("--visual-verification", required=True, type=Path)
    parser.add_argument("--audio-verification", required=True, type=Path)
    parser.add_argument("--pixel-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite output: {args.output}", file=sys.stderr)
        return 2
    try:
        result = join(
            args.main_fact,
            args.gatea_fact,
            args.visual_verification,
            args.audio_verification,
            args.pixel_evidence,
        )
        _write_fresh(args.output, result)
    except F2DirectionPixelJoinError as exc:
        print(f"F2 direction pixel join refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "status": result["pixel_join_status"],
        "point_id": result["point_id"],
        "rejection_reasons": result["rejection_reasons"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
