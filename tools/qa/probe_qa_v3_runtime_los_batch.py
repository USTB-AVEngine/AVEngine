#!/usr/bin/env python3
"""Probe QA-v3 candidate sightlines in one real packaged UE map.

The tool launches a named map once and performs complex ``BlockAll`` traces
from each candidate camera to both controlled-role visual centers at the
candidate's anchor/query frames.  It is a cheap pre-render classifier only;
native target-only pixel truth remains the visibility authority.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.backends.spear_ue.launch import validate_current_production_spear_executable  # noqa: E402
from avengine.backends.spear_ue.research_runtime import launch_external_game_instance  # noqa: E402

SCHEMA = "qa_v3_packaged_runtime_sightline_probe_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def mapping_value(value: Mapping[str, Any], key: str) -> Any:
    folded = key.casefold()
    matches = [item for name, item in value.items() if str(name).casefold() == folded]
    require(len(matches) == 1, f"trace result lacks unique {key}")
    return matches[0]


def required_frames(fact: Mapping[str, Any]) -> list[int]:
    frames = sorted({int(fact["anchor_frame"]), int(fact["query_frame"])})
    require(frames and all(0 <= frame < 75 for frame in frames), "fact frames are invalid")
    return frames


def parse_trace_result(value: Mapping[str, Any]) -> dict[str, Any]:
    hit = bool(mapping_value(value, "ReturnValue"))
    result = {"blocked": hit}
    if hit:
        out_hit = mapping_value(value, "OutHit")
        require(isinstance(out_hit, Mapping), "trace OutHit is invalid")
        location = mapping_value(out_hit, "Location")
        require(isinstance(location, Mapping), "trace hit Location is invalid")
        result["hit_point_ue_cm"] = [
            float(mapping_value(location, axis)) for axis in ("X", "Y", "Z")
        ]
    return result


def load_candidates(inputs_root: Path, actor_center_height_cm: float) -> list[dict[str, Any]]:
    require(inputs_root.is_dir(), f"inputs root is missing: {inputs_root}")
    candidates = []
    for directory in sorted(inputs_root.iterdir()):
        timeline_path = directory / "timeline.json"
        fact_path = directory / "fact_record.json"
        if not timeline_path.is_file() or not fact_path.is_file():
            continue
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        fact = json.loads(fact_path.read_text(encoding="utf-8"))
        frames = required_frames(fact)
        records = []
        for frame_index in frames:
            frame = timeline["frames"][frame_index]
            require(frame["frame_index"] == frame_index, "timeline frame index drift")
            camera = frame["camera"]["translation_ue_cm"]
            actors = {}
            for state in frame["actor_states"]:
                slot = str(state["source_slot_id"])
                position = [float(value) for value in state["translation_ue_cm"]]
                position[2] += actor_center_height_cm
                actors[slot] = position
            require(set(actors) == {"source1", "source2"}, "timeline actor slots differ")
            records.append({
                "frame_index": frame_index,
                "camera_ue_cm": [float(value) for value in camera],
                "actor_centers_ue_cm": actors,
            })
        candidates.append({
            "point_id": directory.name,
            "profile_id": fact["profile_id"],
            "frames": records,
        })
    require(candidates, "inputs root contains no QA-v3 candidates")
    return candidates


def run(args: argparse.Namespace) -> Path:
    executable = validate_current_production_spear_executable(args.spear_executable)
    stage = args.stage_root.resolve()
    require(stage.is_dir(), f"stage root is missing: {stage}")
    try:
        executable.relative_to(stage)
    except ValueError as error:
        raise RuntimeError("spear executable must be inside stage root") from error
    require(args.native_map.startswith("/Game/"), "native map must be a /Game path")
    candidates = load_candidates(args.inputs_root, args.actor_center_height_cm)
    require(not args.output.exists(), f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    instance = launch_external_game_instance(
        spear_executable=executable,
        native_map=args.native_map,
        frame_rate_hz=15,
        rpc_port=args.rpc_port,
        graphics_adapter=args.graphics_adapter,
    )
    per_point = []
    try:
        game = instance.get_game()
        with instance.begin_frame():
            kismet = game.get_unreal_object(uclass="UKismetSystemLibrary")
            for candidate in candidates:
                traces = []
                for frame in candidate["frames"]:
                    start_values = frame["camera_ue_cm"]
                    start = dict(zip(("X", "Y", "Z"), start_values))
                    for slot in ("source1", "source2"):
                        end_values = frame["actor_centers_ue_cm"][slot]
                        end = dict(zip(("X", "Y", "Z"), end_values))
                        raw = kismet.LineTraceSingleByProfile(
                            Start=start,
                            End=end,
                            ProfileName="BlockAll",
                            bTraceComplex=True,
                            ActorsToIgnore=[],
                            DrawDebugType="None",
                            bIgnoreSelf=True,
                            TraceColor={"R": 1.0, "G": 0.0, "B": 0.0, "A": 1.0},
                            TraceHitColor={"R": 0.0, "G": 1.0, "B": 0.0, "A": 1.0},
                            DrawTime=0.0,
                            as_dict=True,
                        )
                        require(isinstance(raw, Mapping), "line trace result is invalid")
                        traces.append({
                            "frame_index": frame["frame_index"],
                            "source_slot_id": slot,
                            "start_ue_cm": start_values,
                            "end_ue_cm": end_values,
                            **parse_trace_result(raw),
                        })
                per_point.append({
                    "point_id": candidate["point_id"],
                    "profile_id": candidate["profile_id"],
                    "status": "clear" if not any(item["blocked"] for item in traces) else "blocked",
                    "traces": traces,
                })
        with instance.end_frame():
            pass
    finally:
        instance.close(force=True)

    report = {
        "schema": SCHEMA,
        "status": "research_pre_render_probe",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": (
            "packaged-UE complex line traces are a cheap candidate pre-screen; "
            "they do not replace same-camera target-only pixel truth"
        ),
        "native_map": args.native_map,
        "actor_center_height_cm": args.actor_center_height_cm,
        "candidate_count": len(per_point),
        "clear_count": sum(item["status"] == "clear" for item in per_point),
        "blocked_count": sum(item["status"] == "blocked" for item in per_point),
        "per_point": per_point,
    }
    output = args.output / "runtime_sightlines.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"QA_V3_RUNTIME_LOS_OK output={output} clear={report['clear_count']} blocked={report['blocked_count']}")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", required=True, type=Path)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--spear-executable", required=True, type=Path)
    parser.add_argument("--native-map", required=True)
    parser.add_argument("--actor-center-height-cm", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rpc-port", type=int, default=39581)
    parser.add_argument("--graphics-adapter", type=int, default=1)
    args = parser.parse_args(argv)
    if not math.isfinite(args.actor_center_height_cm) or args.actor_center_height_cm <= 0.0:
        parser.error("--actor-center-height-cm must be finite and positive")
    if not 1024 <= args.rpc_port <= 65535 or args.graphics_adapter < 0:
        parser.error("invalid RPC port or graphics adapter")
    return args


if __name__ == "__main__":
    run(parse_args())
