#!/usr/bin/env python3
# HISTORICAL TOOL (single-repo closure, 2026-08-21): this script built or
# validates retained strict-two-human evidence recorded against the
# pre-closure transition environment (sibling Habitat fork, sound-spaces,
# SPEAR-lead-b, and multi-repo SPEAR checkouts). The hard-coded absolute
# paths below are a frozen historical record, not current inputs. The current
# production chain runs on the installed runtime prefix and external data
# roots under /data/avengine_external; do not use this tool for new work.
"""Bind the reviewed both-move geometry handoff to A's materializer contract."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
INPUT_SCHEMA = "avengine_native_strict_two_human_dynamic_full75_canary_preflight_v1"
DEFAULT_INPUT = (
    REPOSITORY
    / "examples/qa/native_strict_two_human_both_move_v1_geometry_preflight_v1.json"
)
SOURCE_SUITE = Path(
    "/data/datasets/avengine_workspaces/AVEngine-habitat-native/tmp/m7/"
    "apartment_asset_bound_ue_unique1000_full_20260723_01/"
    "suite_execution_plan.json"
)
SOUND_REGISTRY = Path(
    "/data/jzy/code/SPEAR-lead-b/outputs/lead_b/audio_candidates_v1/"
    "controlled_sound_content_registry_v1.json"
)
CANDIDATE_REVISION = "both_move_v1_0304_0990_equal_arc_v1"
TARGET_AUDIO = {
    "content_id": "cremad_ieo_v1",
    "sound_asset_id": "speech_cremad_1001_ieo_neu_v1",
    "voice_id": "cremad_actor_1001",
    "transcript": "It's eleven o'clock.",
    "speech_sample_count": 25_626,
    "listening_review": "pass",
    "rights_status": "licensed",
}
EXPECTED_SCENARIOS = [
    "human_border_collie__recombined_both_moving_0304",
    "border_collie_human__recombined_both_moving_0990",
]
PATH_METHOD = "equal_arc_interpolation_of_exact_native_human_polyline_v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def adapt(input_path: Path, output: Path) -> Path:
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    source = _load(input_path)
    _require(source.get("schema") == INPUT_SCHEMA, "geometry preflight schema drift")
    rows = source.get("canaries")
    _require(isinstance(rows, list) and len(rows) == 1, "expected one geometry row")
    row = deepcopy(rows[0])
    _require(
        row.get("episode_id") == "strict2h_dynamic_canary_03_both_move_v1"
        and row.get("execution_order") == 3
        and row.get("mechanism") == "both_move"
        and row.get("target_side") == "right"
        and row.get("native_same_scene_pair") is False
        and row.get("native_source_scenario_ids") == EXPECTED_SCENARIOS,
        "both-move candidate identity/provenance drift",
    )
    _require(
        row.get("acoustic_state_expectation")
        == {
            "source_frame_uses": 150,
            "target_unique_rir_states": 75,
            "distractor_unique_rir_states": 75,
            "total_unique_rir_states": 150,
            "exact_rir_required_before_gpu": True,
        },
        "both-move acoustic expectation drift",
    )
    _require(SOURCE_SUITE.is_file(), f"source suite missing: {SOURCE_SUITE}")
    suite = _load(SOURCE_SUITE)
    scenario_ids = {str(item["scenario_id"]) for item in suite["scenarios"]}
    _require(
        set(EXPECTED_SCENARIOS) <= scenario_ids,
        "both native-human source scenarios are not present in the source suite",
    )
    for slot, role_name in (("source1", "target"), ("source2", "distractor")):
        role = row[role_name]
        provenance = role.get("path_provenance", {})
        _require(
            role.get("source_slot_id") == slot
            and len(role.get("root_path_m", [])) == 75
            and len(role.get("per_frame_action_phase", [])) == 75
            and len(role.get("per_frame_anatomical_forward_habitat_world", [])) == 75
            and len(role.get("per_frame_tangent_yaw_habitat_deg", [])) == 75
            and provenance.get("method") == PATH_METHOD
            and provenance.get("counterfactual_cross_scenario_pair") is True
            and provenance.get("endpoints_exact_native_readbacks") is True
            and provenance.get("interior_output_roots_exact_native_frame_readbacks")
            is False
            and provenance.get("native_source_scenario_id")
            == EXPECTED_SCENARIOS[0 if slot == "source1" else 1]
            and provenance.get("output_root_count") == 75
            and provenance.get("output_unique_root_count_at_1mm") == 75,
            f"{slot} derived native-human path provenance drift",
        )
    _require(
        row["target"].get("identity_key") == "M"
        and row["distractor"].get("identity_key") == "F"
        and row["target"].get("sound_asset_id") == "PENDING_CPU_AUDIO_BINDING"
        and row["distractor"].get("voice_policy") == "silent",
        "both-move role/audio handoff drift",
    )
    registry = _load(SOUND_REGISTRY)
    matches = [
        item
        for item in registry.get("assets", [])
        if item.get("sound_asset_id") == TARGET_AUDIO["sound_asset_id"]
    ]
    _require(len(matches) == 1, "controlled target sound did not resolve exactly once")
    sound = matches[0]
    _require(
        sound["content"]["speaker_id"] == TARGET_AUDIO["voice_id"]
        and sound["content"]["statement_id"] == TARGET_AUDIO["content_id"]
        and sound["content"]["transcript"] == TARGET_AUDIO["transcript"]
        and sound["audio"]["sample_count"] == TARGET_AUDIO["speech_sample_count"],
        "controlled target sound metadata drift",
    )

    row["candidate_revision"] = CANDIDATE_REVISION
    row["source_suite"] = str(SOURCE_SUITE)
    row["target"].update(TARGET_AUDIO)
    row["status"] = "pass_cpu_adapter_ready_for_materialization"
    row["gpu_launch_authorized"] = False
    adapted = deepcopy(source)
    adapted["status"] = "pass_cpu_adapter_ready_for_materialization"
    adapted["gpu_launch_authorized"] = False
    adapted["formal_episode_count"] = 0
    adapted["qualification_claim"] = False
    adapted["integration_adapter"] = {
        "schema": "avengine_both_move_cross_scenario_adapter_v1",
        "status": "pass",
        "candidate_revision": CANDIDATE_REVISION,
        "counterfactual_pair": True,
        "native_same_scene_pair": False,
        "native_human_anchor_scenario_ids": EXPECTED_SCENARIOS,
        "interior_root_authority": "derived_equal_arc_interpolation_not_native_readback",
        "audio_contract": "source1_controlled_speech_source2_exact_silence",
        "gpu_launch_authorized": False,
        "formal": False,
        "qualification_claim": False,
    }
    adapted["canaries"] = [row]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(adapted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = adapt(args.input.resolve(), args.output.resolve())
    print(f"BOTH_MOVE_MATERIALIZER_PREFLIGHT_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
