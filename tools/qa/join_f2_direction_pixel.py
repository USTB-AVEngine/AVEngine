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


def _window_geometry_status(
    main_fact: Mapping[str, Any],
    gatea_fact: Mapping[str, Any],
    *,
    role: str,
) -> str | None:
    for fact in (main_fact, gatea_fact):
        for container in (
            fact,
            fact.get("generation_checks"),
            fact.get("truth"),
            fact.get("open"),
        ):
            if not isinstance(container, Mapping):
                continue
            status = _geometry_status(
                container.get("query_visibility_window_geometry"),
                role=role,
            )
            if status is not None:
                return status
    return None


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

    main_window = _window(main, owner="main fact")
    gatea_window = _window(gatea, owner="GateA fact")
    reasons: list[str] = []
    if main_window != gatea_window:
        reasons.append("main_gatea_query_windows_differ")

    geometry = {
        "main": _window_geometry_status(main, gatea, role="main"),
        "gateA": _window_geometry_status(main, gatea, role="gateA"),
    }
    geometry_checks = {
        "main_status": geometry["main"],
        "gateA_status": geometry["gateA"],
        "main_pass": geometry["main"] == "pass",
        "gateA_pass": geometry["gateA"] == "pass",
        "both_pass": (
            geometry["main"] == "pass" and geometry["gateA"] == "pass"
        ),
    }
    if not geometry_checks["main_pass"]:
        reasons.append("main_query_visibility_window_geometry_not_pass")
    if not geometry_checks["gateA_pass"]:
        reasons.append("gateA_query_visibility_window_geometry_not_pass")

    visual = _mapping(_read(visual_path), owner="visual verification")
    audio = _mapping(_read(audio_path), owner="audio verification")
    verifier_checks, verifier_reasons = _verification_checks(visual, audio)
    reasons.extend(verifier_reasons)

    pixel_wrapper = _mapping(_read(pixel_path), owner="pixel evidence")
    truth = _pixel_truth(pixel_wrapper, path=pixel_path)
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
            "query_visibility_window_geometry": geometry_checks,
            "visual_audio_verification": verifier_checks,
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
