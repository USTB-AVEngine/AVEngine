#!/usr/bin/env python3
"""Bind one full native SPEAR capture to the paper-balance QuestionSpec strata."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import canonical_json_sha256, write_json
from avengine.qa.question_protocol import enumerate_episode_specs
from avengine.qa.question_spec import evaluate_question_specs

BASE_BINDER_PATH = REPOSITORY / "tools/qa/bind_native_pixel_fact_episode.py"
BASE_BINDER_SPEC = importlib.util.spec_from_file_location(
    "bind_native_pixel_fact_episode", BASE_BINDER_PATH
)
if BASE_BINDER_SPEC is None or BASE_BINDER_SPEC.loader is None:
    raise RuntimeError(f"cannot import {BASE_BINDER_PATH}")
BASE_BINDER = importlib.util.module_from_spec(BASE_BINDER_SPEC)
BASE_BINDER_SPEC.loader.exec_module(BASE_BINDER)

SCHEMA = "avengine_native_paper_balance_episode_binding_v1"
VARIANTS = {
    "stationary": {
        "episode_id": "border_collie_human__paper_balance_stationary_first_v1",
        "scenario_type": "paper_balance_stationary_first",
        "sound_asset_id": "speech_cremad_1001_ieo_neu_v1",
    },
    "right_entry": {
        "episode_id": "border_collie_human__paper_balance_right_entry_v1",
        "scenario_type": "offscreen_to_onscreen",
        "sound_asset_id": "speech_cremad_1005_tie_neu_v1",
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sound_id(binding: Any) -> str | None:
    if isinstance(binding, str):
        return binding
    if isinstance(binding, Mapping):
        value = binding.get("sound_asset_id") or binding.get("sound_id")
        return value if isinstance(value, str) else None
    return None


def _find_pass(
    specs: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    *,
    question_type: str,
    selectors: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(len(specs) == len(evaluations), "spec/evaluation cardinality drift")
    matches = [
        evaluation
        for spec, evaluation in zip(specs, evaluations)
        if spec.get("question_type") == question_type
        and spec.get("selectors") == selectors
        and evaluation.get("spec_id") == spec.get("spec_id")
    ]
    _require(
        len(matches) == 1,
        f"{question_type} selectors {dict(selectors)!r} matched {len(matches)} cases",
    )
    _require(
        matches[0].get("status") == "pass",
        f"{question_type} did not pass: {matches[0]}",
    )
    return matches[0]


def _answer_value(evaluation: Mapping[str, Any]) -> Any:
    answer = evaluation.get("answer")
    return answer.get("value") if isinstance(answer, Mapping) else None


def _manifest_av_gate(manifest: Mapping[str, Any]) -> None:
    streams = manifest.get("ffprobe", {}).get("streams", [])
    video = [item for item in streams if item.get("codec_type") == "video"]
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    _require(
        len(video) == 1 and video[0].get("nb_frames") == "75",
        "native delivery must contain exactly 75 video frames",
    )
    _require(
        len(audio) == 1 and audio[0].get("channels") == 2,
        "native delivery must contain one stereo audio stream",
    )


def _validate_variant(
    *,
    variant: str,
    facts: Mapping[str, Any],
    pixel_truth: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    event_sound_bindings: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    contract = VARIANTS[variant]
    _require(
        facts.get("episode_id") == contract["episode_id"], "variant Episode id drift"
    )
    request = manifest.get("authoritative_capture_request", {})
    _require(
        request.get("scenario_type") == contract["scenario_type"]
        and request.get("target_source_slot_id") == "source2",
        "native capture request differs from the paper-balance variant",
    )
    _manifest_av_gate(manifest)

    events = facts.get("sound_events")
    _require(
        isinstance(events, Sequence)
        and not isinstance(events, (str, bytes))
        and len(events) == 1,
        "paper-balance Episode must contain exactly one controlled sound event",
    )
    event = events[0]
    _require(
        event.get("source_slot_id") == "source2",
        "source2 must be the only speaking source",
    )
    bound_sound = _sound_id(event_sound_bindings.get(event.get("event_id")))
    _require(
        bound_sound == contract["sound_asset_id"]
        and event.get("sound_asset_id") == contract["sound_asset_id"],
        "controlled sound binding differs from the variant contract",
    )
    source1_frames = pixel_truth["per_instance"]["source1"]["frames"]
    source2_frames = pixel_truth["per_instance"]["source2"]["frames"]
    _require(
        len(source1_frames) == 75
        and len(source2_frames) == 75
        and all(frame.get("visible_pixels", 0) > 0 for frame in source1_frames),
        "the unique silent appearance must remain visibly observable for all 75 frames",
    )

    if variant == "stationary":
        _require(
            all(frame.get("visible_pixels", 0) > 0 for frame in source2_frames),
            "the stationary speaking source must remain visibly observable",
        )
        appearance = _find_pass(
            specs,
            evaluations,
            question_type="appearance_to_speaking",
            selectors={
                "appearance_field": "breed_id",
                "appearance_value": "border_collie",
            },
        )
        first = _find_pass(
            specs,
            evaluations,
            question_type="who_spoke_first",
            selectors={},
        )
        moving = _find_pass(
            specs,
            evaluations,
            question_type="speaking_while_moving",
            selectors={"sound_asset_id": contract["sound_asset_id"]},
        )
        _require(
            _answer_value(appearance) == "no",
            "stationary variant must close appearance_to_speaking=no",
        )
        _require(
            _answer_value(first) == "source2",
            "stationary variant must make source2 the unique first speaker",
        )
        _require(
            _answer_value(moving) == "no",
            "stationary variant must close speaking_while_moving=no",
        )
        moving_truth = facts["tracks"]["instances"]["source2"]["moving"]
        speaking_frames = list(
            range(int(event["start_frame"]), int(event["end_frame"]))
        )
        _require(
            speaking_frames
            and all(not bool(moving_truth[index]) for index in speaking_frames),
            "source2 is not stationary over the complete speaking window",
        )
        return {
            "appearance_to_speaking": "no",
            "who_spoke_first": "source2",
            "speaking_while_moving": "no",
            "silent_visible_instance_id": "source1",
            "speech_frame_indices": speaking_frames,
        }

    entry = _find_pass(
        specs,
        evaluations,
        question_type="offscreen_to_onscreen",
        selectors={"target_instance_id": "source2"},
    )
    content = _find_pass(
        specs,
        evaluations,
        question_type="appearance_to_spoken_content",
        selectors={
            "appearance_field": "sex_or_gender_label",
            "appearance_value": "male",
        },
    )
    transcript = "That is exactly what happened."
    _require(
        _answer_value(entry) == "right",
        "right-entry variant must close offscreen_to_onscreen=right",
    )
    _require(
        _answer_value(content) == transcript
        and event.get("transcript") == transcript
        and event.get("statement_id") == "cremad_tie_v1",
        "right-entry variant must close the second controlled transcript stratum",
    )
    visible = [
        frame
        for frame in source2_frames
        if frame.get("state") in {"visible_clear", "visible_occluded"}
    ]
    _require(visible, "right-entry target never becomes visible")
    first_visible = visible[0]
    width = int(pixel_truth["resolution_hw"][1])
    center = (width - 1) / 2.0
    dead_zone = max(1.0, width * 0.02)
    centroid = first_visible.get("target_centroid_xy_px")
    _require(
        first_visible.get("frame_index") == 23
        and all(
            frame.get("frame_index") == index and frame.get("state") == "out_of_view"
            for index, frame in enumerate(source2_frames[:23])
        )
        and isinstance(centroid, Sequence)
        and not isinstance(centroid, (str, bytes))
        and len(centroid) == 2
        and float(centroid[0]) > center + dead_zone,
        "right-entry native pixel transition or right-side dead-zone gate failed",
    )
    return {
        "offscreen_to_onscreen": "right",
        "appearance_to_spoken_content": transcript,
        "first_visible_frame": 23,
        "first_visible_centroid_x_px": float(centroid[0]),
        "silent_visible_instance_id": "source1",
    }


def _relocated_record(source: Path, destination: Path) -> dict[str, Any]:
    record = BASE_BINDER._file_record(source)
    record["path"] = str(destination.resolve())
    return record


def build(
    *,
    variant: str,
    manifest_path: Path,
    fact_path: Path,
    asset_registry_path: Path,
    output: Path,
) -> Mapping[str, Any]:
    _require(variant in VARIANTS, f"unsupported paper-balance variant: {variant}")
    _require(not output.exists(), f"output already exists: {output}")
    source_facts = _load(fact_path)
    _require(source_facts.get("status") == "pass", "source Facts did not pass")
    registry_inputs = [
        item
        for item in source_facts["provenance"]["inputs"]
        if item.get("role") == "source_asset_runtime_registry"
    ]
    _require(
        len(registry_inputs) == 1
        and BASE_BINDER._sha256(asset_registry_path) == registry_inputs[0]["sha256"],
        "asset registry differs from the registry declared by source Facts",
    )
    manifest, pixel_truth, _artifact_paths = BASE_BINDER._validate_capture(
        manifest_path=manifest_path,
        facts=source_facts,
    )
    facts = deepcopy(source_facts)
    facts["visibility"]["pixel_truth"] = pixel_truth
    facts["provenance"]["inputs"].append(
        {
            "role": "native_spear_pixel_capture_manifest",
            "path": str(manifest_path.resolve()),
            "sha256": BASE_BINDER._sha256(manifest_path),
        }
    )
    jsonschema.validate(
        facts,
        _load(REPOSITORY / "schemas/avengine_qa_fact_table_v1.schema.json"),
    )
    _unused_specs, sound_registry, bindings = BASE_BINDER._question_inputs(
        facts,
        scenario_type=manifest["authoritative_capture_request"]["scenario_type"],
        target_slot="source2",
    )
    specs = enumerate_episode_specs(facts, bindings)
    evaluations = evaluate_question_specs(
        specs,
        facts=facts,
        asset_registry=_load(asset_registry_path),
        sound_registry=sound_registry,
        event_sound_bindings=bindings,
    )
    target_answers = _validate_variant(
        variant=variant,
        facts=facts,
        pixel_truth=pixel_truth,
        specs=specs,
        evaluations=evaluations,
        event_sound_bindings=bindings,
        manifest=manifest,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        payloads = {
            "facts": facts,
            "question_specs": specs,
            "question_evaluations": evaluations,
            "sound_registry": sound_registry,
            "event_sound_bindings": bindings,
        }
        names = {
            "facts": "facts.json",
            "question_specs": "question_specs.json",
            "question_evaluations": "question_evaluations.json",
            "sound_registry": "controlled_sound_registry.json",
            "event_sound_bindings": "event_sound_bindings.json",
        }
        for role, payload in payloads.items():
            write_json(temporary / names[role], payload)
        binding = {
            "schema": SCHEMA,
            "status": "pass",
            "variant": variant,
            "episode_id": facts["episode_id"],
            "pixel_authority": pixel_truth["authority"],
            "target_answers": target_answers,
            "capture_manifest": BASE_BINDER._file_record(manifest_path),
            "source_fact": BASE_BINDER._file_record(fact_path),
            "source_asset_registry": BASE_BINDER._file_record(asset_registry_path),
            "native_artifacts": {
                name: manifest["artifact_records"][name]
                for name in (
                    "native_rgb_binaural",
                    "metric_depth",
                    "pixel_masks",
                    "pixel_visibility_truth",
                    "runtime_readbacks",
                    "normal_object_ids",
                )
            },
            "outputs": {
                role: _relocated_record(
                    temporary / filename,
                    output / filename,
                )
                for role, filename in names.items()
            },
            "question_pass_count": sum(
                item.get("status") == "pass" for item in evaluations
            ),
            "question_candidate_count": len(evaluations),
            "contract_sha256": canonical_json_sha256(
                {
                    "episode_id": facts["episode_id"],
                    "variant": variant,
                    "target_answers": target_answers,
                    "capture_manifest_sha256": BASE_BINDER._sha256(manifest_path),
                    "facts_sha256": BASE_BINDER._sha256(temporary / names["facts"]),
                }
            ),
        }
        write_json(temporary / "manifest.json", binding)
        os.replace(temporary, output)
        return binding
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--fact", required=True, type=Path)
    parser.add_argument(
        "--asset-registry",
        type=Path,
        default=REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(
        variant=args.variant,
        manifest_path=args.capture_manifest.resolve(),
        fact_path=args.fact.resolve(),
        asset_registry_path=args.asset_registry.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
