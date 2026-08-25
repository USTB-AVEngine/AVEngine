#!/usr/bin/env python3
"""Plan one fail-closed native Episode that closes paper answer balance.

The tool only prepares an exact SPEAR/UE replay and controlled AudioProgram.
It does not claim native evidence: RGB, metric depth, target-only passes,
runtime readbacks, RLR audio and the final QuestionSpec answers are all gated
by later native stages.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import (
    canonical_json_sha256,
    write_json,
)
from avengine.timeline.audio_program import (
    bind_audio_program_hash,
    validate_audio_program,
)
from avengine.m6.registry import bind_content_hash
from avengine.m6.sources import validate_sound_asset_registry
from avengine.sensor_rig_trajectory import (
    materialize_sensor_rig_trajectory,
)

AUDIO_BUILDER_PATH = REPOSITORY / "tools/qa/build_native_controlled_audio_program.py"
AUDIO_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "native_controlled_audio_builder", AUDIO_BUILDER_PATH
)
if AUDIO_BUILDER_SPEC is None or AUDIO_BUILDER_SPEC.loader is None:
    raise RuntimeError(f"cannot import {AUDIO_BUILDER_PATH}")
AUDIO_BUILDER = importlib.util.module_from_spec(AUDIO_BUILDER_SPEC)
AUDIO_BUILDER_SPEC.loader.exec_module(AUDIO_BUILDER)

PLAN_SCHEMA_PATH = (
    REPOSITORY / "schemas/avengine_native_paper_balance_episode_plan_v1.schema.json"
)
GPU_REVISION_SCHEMA_PATH = (
    REPOSITORY
    / "schemas/avengine_native_paper_balance_gpu_plan_revision_v1.schema.json"
)
SCHEMA = "avengine_native_paper_balance_episode_recipe_v1"
FRAME_COUNT = 75
TICKS_PER_FRAME = 3200
SOURCE1_HOLD_FRAME = 74
SOURCE2_HOLD_FRAME = 0
RIGHT_ENTRY_SOURCE_FRAME_MAP = (
    tuple(range(74, 29, -2))
    + tuple(range(27, -1, -1))
    + (0,) * 24
)
VARIANT_SCENARIO_TYPE = {
    "stationary_source2_first": "paper_balance_stationary_first",
    "source2_right_entry_second_transcript": "offscreen_to_onscreen",
}
SECOND_SPEECH_ID = "speech_cremad_1005_tie_neu_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, role: str) -> dict[str, Any]:
    return {"role": role, "path": str(path.resolve()), "sha256": _sha256(path)}


def _actor_state(frame: Mapping[str, Any], actor_id: str) -> dict[str, Any]:
    matches = [
        deepcopy(state)
        for state in frame["actor_states"]
        if state["actor_id"] == actor_id
    ]
    _require(len(matches) == 1, f"{actor_id}: source actor state must resolve once")
    return matches[0]


def _reflect_xz_about_camera_forward(
    position_m: Sequence[float],
    *,
    camera_position_m: Sequence[float],
    camera_forward_world: Sequence[float],
) -> list[float]:
    point = np.asarray(position_m, dtype=np.float64)
    camera = np.asarray(camera_position_m, dtype=np.float64)
    forward = np.asarray(camera_forward_world, dtype=np.float64)[[0, 2]]
    forward /= np.linalg.norm(forward)
    delta = point[[0, 2]] - camera[[0, 2]]
    reflected = 2.0 * float(np.dot(delta, forward)) * forward - delta
    return [
        float(camera[0] + reflected[0]),
        float(point[1]),
        float(camera[2] + reflected[1]),
    ]


def _project_horizontal_offset_fraction(
    position_m: Sequence[float],
    *,
    camera_position_m: Sequence[float],
    camera_forward_world: Sequence[float],
    horizontal_fov_deg: float,
) -> float:
    point = np.asarray(position_m, dtype=np.float64)
    camera = np.asarray(camera_position_m, dtype=np.float64)
    forward = np.asarray(camera_forward_world, dtype=np.float64)[[0, 2]]
    forward /= np.linalg.norm(forward)
    right = np.asarray([forward[1] * -1.0, forward[0]], dtype=np.float64)
    delta = point[[0, 2]] - camera[[0, 2]]
    depth = float(np.dot(delta, forward))
    _require(depth > 0.0, "target root is behind the planned camera")
    lateral = float(np.dot(delta, right))
    return lateral / (depth * math.tan(math.radians(horizontal_fov_deg) / 2.0))


def _idle_state(
    state: dict[str, Any], actor: Mapping[str, Any], *, frame_index: int
) -> dict[str, Any]:
    state["action_id"] = "idle"
    state["action_phase"] = 0.0
    state["action_time_ticks"] = frame_index * TICKS_PER_FRAME
    state["ue_animation"] = actor["idle_animation"]
    return state


def _set_position(state: dict[str, Any], position_m: Sequence[float]) -> None:
    state["translation_m"] = [float(value) for value in position_m]
    state["translation_ue_cm"] = [
        float(position_m[0]) * 100.0,
        float(position_m[2]) * 100.0,
        float(position_m[1]) * 100.0,
    ]


def _speech_sound_record(
    record: Mapping[str, Any], *, rights_evidence_sha256: str
) -> dict[str, Any]:
    content = record["content"]
    audio = record["audio"]
    return {
        "sound_asset_id": record["sound_asset_id"],
        "revision": "v1",
        "semantic_sound_class": "human_speech",
        "taxonomy_path": ["human", "voice", "speech"],
        "instance_lineage_id": content["speaker_id"],
        "dry_audio": {
            "uri": audio["uri"],
            "sha256": audio["sha256"],
            "sample_rate_hz": audio["sample_rate_hz"],
            "channel_count": audio["channel_count"],
            "sample_count": audio["sample_count"],
        },
        "normalization_policy": {"mode": "preserve", "target_dbfs": None},
        "allowed_transforms": ["crop", "gain", "zero_pad"],
        "permitted_event_usage": ["intermittent_events"],
        "tags": sorted(set(content["content_tags"] + [content["species"]])),
        "provenance": {
            "origin": "lead_b_controlled_sound_content_registry_v1",
            "license": None,
            "rights_status": "review_required",
            "rights_evidence_sha256": rights_evidence_sha256,
        },
        "admissibility": "research",
    }


def _controlled_audio(
    *,
    variant: str,
    controlled_registry: Mapping[str, Any],
    controlled_registry_path: Path,
    runtime_registry: Mapping[str, Any],
    runtime_registry_path: Path,
) -> dict[str, Any]:
    result = AUDIO_BUILDER.build_contracts(
        controlled_registry=controlled_registry,
        controlled_registry_path=controlled_registry_path,
        runtime_registry=runtime_registry,
        runtime_registry_path=runtime_registry_path,
    )
    speech_id = (
        AUDIO_BUILDER.SPEECH_SOUND_ID
        if variant == "stationary_source2_first"
        else SECOND_SPEECH_ID
    )
    speech = AUDIO_BUILDER._controlled_asset(controlled_registry, speech_id)
    media_path = controlled_registry_path.parent / "media" / f"{speech_id}.wav"
    AUDIO_BUILDER._validate_wave(media_path, speech["audio"])
    sounds = deepcopy(result["sound_asset_registry"])
    if speech_id not in {
        item["sound_asset_id"] for item in sounds["sound_assets"]
    }:
        sounds["sound_assets"].append(
            _speech_sound_record(
                speech, rights_evidence_sha256=_sha256(controlled_registry_path)
            )
        )
        sounds["sound_assets"].sort(key=lambda item: item["sound_asset_id"])
    selected_sound = next(
        item for item in sounds["sound_assets"] if item["sound_asset_id"] == speech_id
    )
    selected_sound["permitted_event_usage"] = sorted(
        set(selected_sound["permitted_event_usage"] + ["one_active_of_n"])
    )
    sounds.pop("content_sha256", None)
    sounds = bind_content_hash(sounds)
    start_sample = 8000 if variant == "stationary_source2_first" else 32000
    end_sample = start_sample + int(speech["audio"]["sample_count"])
    _require(end_sample <= 80000, "controlled speech does not fit the formal timeline")
    program = bind_audio_program_hash(
        {
            "schema": "avengine_m6_audio_program_v1",
            "program_id": f"lead_a_native_{variant}_audio_v1",
            "revision": "v1",
            "mode": "one_active_of_n",
            "timeline": {
                "time_base_hz": 48000,
                "ticks_per_frame": TICKS_PER_FRAME,
                "video_fps": 15,
                "frame_count": FRAME_COUNT,
                "sample_rate_hz": 16000,
                "ticks_per_sample": 3,
                "sample_count": 80000,
            },
            "candidate_source_endpoint_ids": [
                "lead_a_source1_muzzle",
                "lead_a_source2_mouth",
            ],
            "events": [
                AUDIO_BUILDER._event(
                    event_id="source2_speech_000",
                    endpoint_id="lead_a_source2_mouth",
                    sound_id=speech_id,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    source_start=0,
                    gain=0.18,
                )
            ],
            "source_specific_stems": True,
            "admission_state": "research",
        }
    )
    sound_errors = validate_sound_asset_registry(sounds)
    program_errors = validate_audio_program(
        program,
        source_endpoint_registry=result["source_endpoint_registry"],
        sound_asset_registry=sounds,
    )
    _require(not sound_errors, "; ".join(sound_errors))
    _require(not program_errors, "; ".join(program_errors))
    result["sound_asset_registry"] = sounds
    result["audio_program"] = program
    result["sound_audio_paths"][speech_id] = str(media_path.resolve())
    result["controlled_content"]["source2"] = {
        "sound_asset_id": speech_id,
        "statement_id": speech["content"]["statement_id"],
        "transcript": speech["content"]["transcript"],
        "language": speech["content"]["language"],
    }
    return result


def build(
    *,
    plan_path: Path,
    source_fact_path: Path,
    source_suite_path: Path,
    source_pixel_truth_path: Path,
    controlled_registry_path: Path,
    runtime_registry_path: Path,
    gpu_revision_path: Path | None,
    variant: str,
    output: Path,
) -> dict[str, Path]:
    plan = _load(plan_path)
    jsonschema.Draft202012Validator(_load(PLAN_SCHEMA_PATH)).validate(plan)
    configs = [item for item in plan["episodes"] if item["variant"] == variant]
    _require(len(configs) == 1, f"variant {variant!r} must resolve once")
    config = configs[0]
    gpu_policy = deepcopy(plan["gpu_policy"])
    gpu_revision = None
    if gpu_revision_path is not None:
        gpu_revision = _load(gpu_revision_path)
        jsonschema.Draft202012Validator(_load(GPU_REVISION_SCHEMA_PATH)).validate(
            gpu_revision
        )
        _require(
            gpu_revision["superseded_gpu_policy"] == plan["gpu_policy"],
            "GPU revision does not supersede the selected base plan exactly",
        )
        gpu_policy = deepcopy(gpu_revision["active_gpu_policy"])
    source_fact = _load(source_fact_path)
    source_suite = _load(source_suite_path)
    source_truth = _load(source_pixel_truth_path)
    _require(source_fact.get("time", {}).get("frame_count") == FRAME_COUNT, "Fact frame drift")
    _require(len(source_suite.get("scenarios", [])) == 1, "source suite must contain one scenario")
    source_scenario = source_suite["scenarios"][0]
    _require(source_scenario["scenario_id"] == source_fact["episode_id"], "source Episode drift")
    source_frames = source_scenario["plan"]["frames"]
    _require(len(source_frames) == FRAME_COUNT, "source suite frame drift")
    _require(source_truth.get("frame_indices") == list(range(FRAME_COUNT)), "source truth frame drift")
    actors = {item["actor_id"]: item for item in source_scenario["plan"]["actors"]}
    _require(set(actors) == {"source1_actor", "source2_actor"}, "actor set drift")

    listener = source_fact["listener"]
    camera_position = listener["position_m"]
    camera_forward = listener["forward_world"]
    hfov = float(source_fact["visibility"]["hfov_degrees"])
    sensor_rig = materialize_sensor_rig_trajectory(
        trajectory_id=f"{config['episode_id']}__sensor_rig",
        program={
            "kind": "HOLD",
            "position_m": camera_position,
            "yaw_deg": float(listener["yaw_deg"]),
        },
    )

    source1_state = _actor_state(source_frames[SOURCE1_HOLD_FRAME], "source1_actor")
    source1_root = deepcopy(source1_state["translation_m"])
    source1_emitter = deepcopy(
        source_fact["tracks"]["instances"]["source1"]["emitter_position_m"][
            SOURCE1_HOLD_FRAME
        ]
    )
    frame_map = (
        (SOURCE2_HOLD_FRAME,) * FRAME_COUNT
        if variant == "stationary_source2_first"
        else RIGHT_ENTRY_SOURCE_FRAME_MAP
    )
    _require(len(frame_map) == FRAME_COUNT, "internal source frame map drift")
    new_frames: list[dict[str, Any]] = []
    roots: dict[str, list[list[float]]] = {"source1": [], "source2": []}
    emitters: dict[str, list[list[float]]] = {"source1": [], "source2": []}
    for frame_index, source_index in enumerate(frame_map):
        camera_state = deepcopy(source_frames[0]["camera_state"])
        camera_state["frame_index"] = frame_index
        camera_state["pts_ticks"] = frame_index * TICKS_PER_FRAME
        first = _idle_state(
            deepcopy(source1_state), actors["source1_actor"], frame_index=frame_index
        )
        second = _actor_state(source_frames[source_index], "source2_actor")
        second_root = deepcopy(second["translation_m"])
        second_emitter = deepcopy(
            source_fact["tracks"]["instances"]["source2"]["emitter_position_m"][
                source_index
            ]
        )
        if variant == "source2_right_entry_second_transcript":
            second_root = _reflect_xz_about_camera_forward(
                second_root,
                camera_position_m=camera_position,
                camera_forward_world=camera_forward,
            )
            second_emitter = _reflect_xz_about_camera_forward(
                second_emitter,
                camera_position_m=camera_position,
                camera_forward_world=camera_forward,
            )
            _set_position(second, second_root)
            if frame_index >= 51:
                _idle_state(second, actors["source2_actor"], frame_index=frame_index)
        else:
            _idle_state(second, actors["source2_actor"], frame_index=frame_index)
        new_frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * TICKS_PER_FRAME,
                "camera_state": camera_state,
                "actor_states": [first, second],
            }
        )
        roots["source1"].append(deepcopy(source1_root))
        roots["source2"].append(deepcopy(second_root))
        emitters["source1"].append(deepcopy(source1_emitter))
        emitters["source2"].append(deepcopy(second_emitter))

    source2_displacement = max(
        float(np.linalg.norm(np.asarray(point) - np.asarray(roots["source2"][0])))
        for point in roots["source2"]
    )
    retained_source1 = source_truth["per_instance"]["source1"]["frames"][
        SOURCE1_HOLD_FRAME
    ]
    retained_source2_hold = source_truth["per_instance"]["source2"]["frames"][
        SOURCE2_HOLD_FRAME
    ]
    _require(retained_source1["state"] == "visible_clear", "silent source1 seed is not visible_clear")
    _require(retained_source2_hold["state"] in {"visible_clear", "visible_occluded"}, "source2 hold seed is not visible")
    preflight: dict[str, Any] = {
        "schema": "avengine_native_paper_balance_preflight_v1",
        "status": "pass_pending_sparse_native_gate",
        "qualification_claim": False,
        "variant": variant,
        "episode_id": config["episode_id"],
        "retained_native_seed_visibility": {
            "source1_hold": retained_source1,
            "source2_hold": retained_source2_hold,
            "authority": source_truth["authority"],
        },
        "source1_event_count": config["source1_event_count"],
        "gpu_policy": gpu_policy,
        "native_recapture_required": True,
    }
    if variant == "stationary_source2_first":
        _require(source2_displacement <= 1.0e-12, "stationary source2 root drift")
        preflight["stationary_source2"] = {
            "maximum_root_displacement_m": source2_displacement,
            "speech_window_requires_all_moving_false": True,
        }
        sparse_frames = [0, 8, 32, 74]
    else:
        _require(
            all(
                source_truth["per_instance"]["source2"]["frames"][index]["state"]
                == "out_of_view"
                for index in range(30, 75)
            ),
            "retained source2 out-of-view suffix drift",
        )
        _require(
            source_truth["per_instance"]["source2"]["frames"][27]["state"]
            in {"visible_clear", "visible_occluded"},
            "retained source2 frame27 is not visible",
        )
        offset = _project_horizontal_offset_fraction(
            emitters["source2"][23],
            camera_position_m=camera_position,
            camera_forward_world=camera_forward,
            horizontal_fov_deg=hfov,
        )
        _require(offset > 0.02, "planned first entry does not clear the right dead zone")
        preflight["right_entry_projection"] = {
            "last_planned_out_frame": 22,
            "first_planned_visible_frame": 23,
            "projected_horizontal_offset_fraction": offset,
            "required_minimum_fraction": 0.02,
            "reflection_requires_native_recapture": True,
        }
        sparse_frames = [0, 22, 23, 24, 50, 51, 74]

    bank_episode = {
        "episode_id": config["episode_id"],
        "motion_case": variant,
        "source_center_paths_m": emitters,
        "source_root_paths_m": roots,
        "statistics": {
            "source_episode_id": source_fact["episode_id"],
            "source_frame_index_map": list(frame_map),
            "native_recapture_required": True,
            "paper_required_answers": config["required_answers"],
        },
    }
    trajectory_bank = {
        "schema": "avengine_room_trajectory_bank_v2",
        "seed": 20260810,
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": 15,
        "seconds_per_episode": 5,
        "source_slots": ["source1", "source2"],
        "episode_count": 1,
        "motion_case_counts": {variant: 1},
        "claim_boundary": "native replay recipe pending sparse and full SPEAR capture",
        "path_semantics": {
            "source_center_paths_m": "asset-bound world emitter points",
            "source_root_paths_m": "asset-bound actor roots",
        },
        "semantics": "retained native 0323 actor states with explicit paper-balance replay",
        "episodes": [bank_episode],
    }
    instances = {item["instance_id"]: item for item in source_fact["instances"]}
    binding_report = {
        "schema": "avengine_asset_emitter_scenario_report_v1",
        "status": "pass",
        "method": "retained_0323_exact_asset_emitter_binding",
        "scenario_count": 1,
        "scenarios": [
            {
                "trajectory_episode_id": config["episode_id"],
                "output_episode_id": config["episode_id"],
                "binding_report": {
                    "schema": "avengine_asset_emitter_binding_report_v1",
                    "status": "pass",
                    "method": "explicit_asset_bound_emitter_paths",
                    "episode_count": 1,
                    "listener_position_m": deepcopy(camera_position),
                    "bindings": [
                        {
                            "source_slot_id": slot,
                            "asset_id": actors[f"{slot}_actor"]["asset_id"],
                            "asset_revision": instances[slot]["registry"][
                                "asset_revision"
                            ],
                            "semantic_anchor_id": instances[slot]["emitter"][
                                "anchor_id"
                            ],
                            "emitter_offset_m": deepcopy(
                                instances[slot]["emitter"]["offset_m"]
                            ),
                            "offset_space": instances[slot]["emitter"]["offset_space"],
                            "local_anatomical_forward_axis": actors[f"{slot}_actor"][
                                "habitat_local_anatomical_forward_axis"
                            ],
                        }
                        for slot in ["source1", "source2"]
                    ],
                    "mouth_animation_required": False,
                    "skeleton_lookup_required": False,
                },
            }
        ],
    }
    scenario = deepcopy(source_scenario)
    scenario["scenario_id"] = config["episode_id"]
    scenario["variant_id"] = variant
    scenario["scenario_directory"] = config["episode_id"]
    scenario["plan"]["frames"] = new_frames
    scenario["plan"]["camera"] = {
        **deepcopy(source_scenario["plan"]["camera"]),
        "sensor_rig_trajectory_id": sensor_rig["trajectory_id"],
        "dynamic": False,
    }
    scenario.pop("dynamic_camera_upgrade", None)
    scenario["static_camera_upgrade"] = {
        "schema": sensor_rig["schema"],
        "trajectory_id": sensor_rig["trajectory_id"],
        "pose_hash": sensor_rig["frames"][0]["pose_hash"],
    }
    scenario["authoritative_capture_request"] = {
        "request_id": f"{config['episode_id']}__native_capture",
        "episode_id": config["episode_id"],
        "scenario_type": VARIANT_SCENARIO_TYPE[variant],
        "target_source_slot_id": "source2",
        "fact_path": "PENDING_FACT_COMPILATION",
        "fact_sha256": "PENDING_FACT_COMPILATION",
    }
    suite = {
        **deepcopy(source_suite),
        "scenarios": [scenario],
        "camera_upgrade": {
            "schema": "avengine_static_spear_suite_camera_upgrade_v1",
            "source_suite": str(source_suite_path.resolve()),
            "source_suite_sha256": _sha256(source_suite_path),
            "sensor_rig_trajectory_id": sensor_rig["trajectory_id"],
            "qualification_claim": False,
        },
    }

    controlled = _controlled_audio(
        variant=variant,
        controlled_registry=_load(controlled_registry_path),
        controlled_registry_path=controlled_registry_path,
        runtime_registry=_load(runtime_registry_path),
        runtime_registry_path=runtime_registry_path,
    )
    output.mkdir(parents=True, exist_ok=False)
    audio_root = output / "controlled_audio_program"
    audio_root.mkdir()
    paths = {
        "trajectory_bank": output / "trajectory_bank.json",
        "asset_emitter_binding_report": output / "asset_emitter_binding_report.json",
        "sensor_rig_trajectory": output / "sensor_rig_trajectory.json",
        "suite": output / "suite_execution_plan.pending_fact.json",
        "preflight": output / "preflight.json",
        "sparse_gate_request": output / "sparse_native_gate_request.json",
        "source_endpoint_registry": audio_root / "source_endpoint_registry.json",
        "sound_asset_registry": audio_root / "sound_asset_registry.json",
        "audio_program": audio_root / "audio_program.json",
        "controlled_audio_binding": audio_root / "controlled_audio_binding.json",
    }
    write_json(paths["trajectory_bank"], trajectory_bank)
    write_json(paths["asset_emitter_binding_report"], binding_report)
    write_json(paths["sensor_rig_trajectory"], sensor_rig)
    write_json(paths["suite"], suite)
    write_json(paths["preflight"], preflight)
    write_json(
        paths["sparse_gate_request"],
        {
            "schema": "avengine_native_paper_balance_sparse_gate_request_v1",
            "status": "ready",
            "qualification_claim": False,
            "episode_id": config["episode_id"],
            "variant": variant,
            "frame_indices": sparse_frames,
            **gpu_policy,
        },
    )
    for key in ["source_endpoint_registry", "sound_asset_registry", "audio_program"]:
        write_json(paths[key], controlled.pop(key))
    write_json(paths["controlled_audio_binding"], controlled)
    recipe = {
        "schema": SCHEMA,
        "status": "prepared_pending_sparse_native_gate",
        "qualification_claim": False,
        "episode_id": config["episode_id"],
        "variant": variant,
        "scenario_type": VARIANT_SCENARIO_TYPE[variant],
        "target_source_slot_id": "source2",
        "inputs": [
            _file_record(plan_path, role="paper_balance_plan"),
            _file_record(source_fact_path, role="retained_source_fact"),
            _file_record(source_suite_path, role="retained_source_native_suite"),
            _file_record(source_pixel_truth_path, role="retained_native_pixel_truth"),
            _file_record(controlled_registry_path, role="controlled_sound_content_registry"),
            _file_record(runtime_registry_path, role="source_asset_runtime_registry"),
        ]
        + (
            []
            if gpu_revision_path is None
            else [_file_record(gpu_revision_path, role="gpu_plan_revision")]
        ),
        "outputs": {
            key: _file_record(path, role=key) for key, path in paths.items()
        },
    }
    recipe["recipe_identity_sha256"] = canonical_json_sha256(recipe)
    paths["recipe"] = output / "recipe.json"
    write_json(paths["recipe"], recipe)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPOSITORY / "examples/qa/native_paper_balance_episode_plan_v1.json",
    )
    parser.add_argument("--source-fact", type=Path, required=True)
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--source-pixel-truth", type=Path, required=True)
    parser.add_argument("--controlled-registry", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--gpu-revision", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANT_SCENARIO_TYPE), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = build(
        plan_path=args.plan.resolve(),
        source_fact_path=args.source_fact.resolve(),
        source_suite_path=args.source_suite.resolve(),
        source_pixel_truth_path=args.source_pixel_truth.resolve(),
        controlled_registry_path=args.controlled_registry.resolve(),
        runtime_registry_path=args.runtime_registry.resolve(),
        gpu_revision_path=(
            None if args.gpu_revision is None else args.gpu_revision.resolve()
        ),
        variant=args.variant,
        output=args.output.resolve(),
    )
    print(
        "NATIVE_PAPER_BALANCE_RECIPE_OK "
        f"variant={args.variant} recipe={paths['recipe']} preflight={paths['preflight']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
