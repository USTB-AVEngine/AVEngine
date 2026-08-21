#!/usr/bin/env python3
"""Bind one full native SPEAR pixel capture to Facts and QuestionSpecs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import jsonschema
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import canonical_json_sha256, write_json  # noqa: E402
from avengine.qa.pixel_visibility import (  # noqa: E402
    PIXEL_VISIBILITY_DEPTH_AUTHORITY,
    bind_pixel_visibility_truth,
)
from avengine.qa.question_spec import evaluate_question_specs  # noqa: E402


FINALIZER_PATH = REPOSITORY / "tools/qa/finalize_native_pixel_artifacts.py"
FINALIZER_SPEC = importlib.util.spec_from_file_location(
    "finalize_native_pixel_artifacts", FINALIZER_PATH
)
if FINALIZER_SPEC is None or FINALIZER_SPEC.loader is None:
    raise RuntimeError(f"cannot import {FINALIZER_PATH}")
FINALIZER = importlib.util.module_from_spec(FINALIZER_SPEC)
FINALIZER_SPEC.loader.exec_module(FINALIZER)

SCHEMA = "avengine_native_pixel_fact_episode_binding_v1"
OCCLUSION_ANCHOR_FRAME = 45


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


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _expected_ue_cm(position_m: Sequence[float]) -> np.ndarray:
    return np.asarray(
        [position_m[0] * 100.0, position_m[2] * 100.0, position_m[1] * 100.0],
        dtype=np.float64,
    )


def _validate_capture(
    *, manifest_path: Path, facts: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    original_manifest = _load(manifest_path)
    finalized = FINALIZER.finalize(manifest_path)
    _require(
        original_manifest.get("artifact_records") == finalized["artifact_records"]
        and original_manifest.get("sha256") == finalized["sha256"],
        "capture artifact inventory was not finalized before binding",
    )
    manifest = dict(finalized)
    request = manifest.get("authoritative_capture_request")
    _require(
        isinstance(request, Mapping)
        and request.get("episode_id") == facts["episode_id"]
        and request.get("fact_sha256") == _sha256(Path(request["fact_path"])),
        "capture request does not bind the authoritative Fact file",
    )
    _require(
        manifest.get("scenario_id") == facts["episode_id"]
        and manifest.get("native_pixel_fact_binding_claim") is True,
        "capture Episode identity or native claim drift",
    )
    frame_contract = manifest["frame_contract"]
    _require(
        frame_contract.get("frame_count") == 75
        and frame_contract.get("formal_episode_frame_count") == 75
        and frame_contract.get("captured_frame_indices") == list(range(75)),
        "Fact binding requires a complete 75-frame native capture",
    )
    paths = {
        name: Path(path).resolve() for name, path in manifest["artifacts"].items()
    }
    truth = _load(paths["pixel_visibility_truth"])
    _require(
        truth.get("authority") == PIXEL_VISIBILITY_DEPTH_AUTHORITY
        and truth.get("frame_indices") == list(range(75))
        and truth.get("camera_pose_ids") == frame_contract["camera_pose_ids"],
        "native truth authority/frame/camera contract drift",
    )
    bound_truth = bind_pixel_visibility_truth(
        truth,
        expected_instance_ids=list(facts["tracks"]["instances"]),
        expected_frame_count=75,
        expected_resolution_hw=facts["visibility"]["resolution_hw"],
        expected_camera_pose_ids=frame_contract["camera_pose_ids"],
    )

    readbacks = _load(paths["runtime_readbacks"])
    _require(
        len(readbacks.get("normal", [])) == 75
        and set(readbacks.get("target_only", {}))
        == set(facts["tracks"]["instances"]),
        "native runtime readbacks are incomplete",
    )
    for pass_name, records in [
        ("normal", readbacks["normal"]),
        *sorted(readbacks["target_only"].items()),
    ]:
        _require(len(records) == 75, f"{pass_name} readback count drift")
        for frame_index, record in enumerate(records):
            _require(
                record["camera"]["expected_pose_hash"]
                == frame_contract["camera_pose_ids"][frame_index],
                f"{pass_name} camera pose drift at frame {frame_index}",
            )
            for slot_id, track in facts["tracks"]["instances"].items():
                observed = np.asarray(
                    record["actors"][f"{slot_id}_actor"]["location_cm"],
                    dtype=np.float64,
                )
                expected = _expected_ue_cm(track["root_position_m"][frame_index])
                _require(
                    float(np.max(np.abs(observed - expected))) <= 1.0e-6,
                    f"{pass_name}/{slot_id} root drift at frame {frame_index}",
                )

    comparison = truth["depth_comparison"]
    background = float(comparison["target_only_background_depth_m"])
    absolute = float(comparison["absolute_tolerance_m"])
    relative = float(comparison["relative_tolerance"])
    with np.load(paths["metric_depth"]) as depth_payload, np.load(
        paths["pixel_masks"]
    ) as mask_payload:
        _require(
            set(depth_payload.files) == FINALIZER.DEPTH_KEYS
            and set(mask_payload.files) == FINALIZER.MASK_KEYS,
            "native depth/mask arrays differ from the exact contract",
        )
        normal = depth_payload["normal_depth_m"]
        expected_shape = (75, *facts["visibility"]["resolution_hw"])
        _require(normal.shape == expected_shape, "normal depth shape drift")
        semantic = mask_payload["depth_derived_modal_semantic"]
        recomputed = np.zeros(expected_shape, dtype=np.uint8)
        best = np.full(expected_shape, np.inf, dtype=np.float32)
        for slot_id in sorted(facts["tracks"]["instances"]):
            semantic_id = int(truth["per_instance"][slot_id]["semantic_id"])
            target = depth_payload[f"target_only_{slot_id}_depth_m"]
            footprint = target < background
            residual = np.abs(normal.astype(np.float32) - target.astype(np.float32))
            visible = footprint & (residual <= absolute + relative * target)
            wins = visible & (residual < best)
            recomputed[wins] = semantic_id
            best[wins] = residual[wins]
            retained_visible = mask_payload[f"modal_visible_{slot_id}"]
            retained_target = mask_payload[f"target_only_{slot_id}"] == semantic_id
            _require(
                np.array_equal(retained_visible, semantic == semantic_id)
                and np.array_equal(retained_target, footprint)
                and not np.any(retained_visible & ~retained_target),
                f"{slot_id} retained masks fail exact depth/subset binding",
            )
            for frame_index, frame in enumerate(
                truth["per_instance"][slot_id]["frames"]
            ):
                _require(
                    frame["visible_pixels"]
                    == int(np.count_nonzero(retained_visible[frame_index]))
                    and frame["target_pixels"]
                    == int(np.count_nonzero(retained_target[frame_index])),
                    f"{slot_id} pixel count drift at frame {frame_index}",
                )
        _require(
            np.array_equal(semantic, recomputed),
            "retained modal semantics differ from metric-depth derivation",
        )
    return manifest, bound_truth, paths


def _question_inputs(
    facts: Mapping[str, Any], *, scenario_type: str, target_slot: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_slot: dict[str, list[Mapping[str, Any]]] = {}
    for event in facts["sound_events"]:
        by_slot.setdefault(event["source_slot_id"], []).append(event)
    sound_assets = []
    bindings: dict[str, Any] = {}
    sound_id_by_slot: dict[str, str] = {}
    for slot_id, events in sorted(by_slot.items()):
        dry = events[0]["dry_variant"]
        species = events[0]["sound_class"]["species_id"]
        declared_sound_ids = {event.get("sound_asset_id") for event in events}
        declared_sound_ids.discard(None)
        _require(
            len(declared_sound_ids) <= 1,
            f"{slot_id} uses multiple controlled sound_asset_id values",
        )
        sound_id = (
            next(iter(declared_sound_ids))
            if declared_sound_ids
            else f"controlled_{slot_id}_{species}_v1"
        )
        sound_id_by_slot[slot_id] = sound_id
        sound_record = {
            "sound_asset_id": sound_id,
            "species": species,
            "path": dry["input_path"],
            "sha256": dry["input_sha256"],
            "admissibility": "research",
        }
        statements = {
            (
                event.get("statement_id"),
                event.get("transcript"),
                event.get("language"),
            )
            for event in events
            if any(
                event.get(field) is not None
                for field in ("statement_id", "transcript", "language")
            )
        }
        _require(
            len(statements) <= 1,
            f"{slot_id} uses multiple controlled statements",
        )
        if statements:
            statement_id, transcript, language = next(iter(statements))
            _require(
                all(
                    isinstance(value, str) and value
                    for value in (statement_id, transcript, language)
                ),
                f"{slot_id} has an incomplete controlled statement",
            )
            sound_record.update(
                {
                    "statement_id": statement_id,
                    "transcript": transcript,
                    "language": language,
                }
            )
        sound_assets.append(sound_record)
        for event in events:
            _require(
                event["dry_variant"]["input_sha256"] == dry["input_sha256"],
                f"{slot_id} uses multiple dry variants",
            )
            bindings[event["event_id"]] = {"sound_asset_id": sound_id}

    target_instance = target_slot
    specs = [
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-007",
            "question_type": "offscreen_to_onscreen",
            "selectors": {"target_instance_id": target_instance},
        },
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-009",
            "question_type": "reappeared_after_occlusion",
            "selectors": {"target_instance_id": target_instance},
        },
    ]
    if scenario_type in {
        "occlusion_to_reappearance",
        "full_occlusion_to_reappearance",
    }:
        frames = facts["visibility"]["pixel_truth"]["per_instance"][target_instance][
            "frames"
        ]
        active_events = by_slot[target_slot]
        eligible = [
            frame["frame_index"]
            for frame in frames
            if frame["state"] in {"visible_occluded", "fully_occluded"}
            and any(
                event["start_frame"] <= frame["frame_index"] < event["end_frame"]
                for event in active_events
            )
        ]
        _require(eligible, "no pixel-occluded speaking frame is available")
        fully_eligible = [
            frame["frame_index"]
            for frame in frames
            if frame["state"] == "fully_occluded"
            and any(
                event["start_frame"] <= frame["frame_index"] < event["end_frame"]
                for event in active_events
            )
        ]
        if scenario_type == "full_occlusion_to_reappearance":
            _require(
                fully_eligible,
                "full-occlusion scenario has no fully_occluded speaking frame",
            )
            occlusion_anchor_frame = fully_eligible[0]
        else:
            _require(
                OCCLUSION_ANCHOR_FRAME in eligible,
                f"required speaking occlusion anchor frame {OCCLUSION_ANCHOR_FRAME} is unavailable",
            )
            occlusion_anchor_frame = OCCLUSION_ANCHOR_FRAME
        specs.append(
            {
                "schema": "avengine_qa_question_spec_v1",
                "spec_id": "QS-008",
                "question_type": "occlusion_while_speaking",
                "selectors": {
                    "sound_asset_id": sound_id_by_slot[target_slot],
                    "frame_index": occlusion_anchor_frame,
                },
            }
        )
        if scenario_type == "occlusion_to_reappearance":
            specs.append(
                {
                    "schema": "avengine_qa_question_spec_v1",
                    "spec_id": "QS-011",
                    "question_type": "became_clear_after_partial_occlusion",
                    "selectors": {"target_instance_id": target_instance},
                }
            )
        evidence = facts["visibility"].get("occluder_evidence")
        records = evidence.get("frame_records") if isinstance(evidence, Mapping) else []
        occluder_frames = [
            record
            for record in records
            if record.get("target_instance_id") == target_instance
            and record.get("occluder_instance_ids")
            and any(
                event["start_frame"] <= record["frame_index"] < event["end_frame"]
                for event in active_events
            )
        ]
        anchor_records = [
            record
            for record in occluder_frames
            if record["frame_index"] == occlusion_anchor_frame
        ]
        if scenario_type == "occlusion_to_reappearance":
            _require(
                len(anchor_records) == 1,
                f"frame {occlusion_anchor_frame} must have exactly one unique native static occluder record",
            )
        # A full target can legitimately be covered by two static objects.  In
        # that case the pixel truth remains a valid full-occlusion positive,
        # but a singular "which object" question has no unique answer and must
        # not be emitted.  Partial-occlusion canaries retain the older strict
        # single-occluder requirement.
        if len(anchor_records) == 1:
            specs.append(
                {
                    "schema": "avengine_qa_question_spec_v1",
                    "spec_id": "QS-010",
                    "question_type": "occluder_identity",
                    "selectors": {
                        "target_instance_id": target_instance,
                        "frame_index": occlusion_anchor_frame,
                    },
                }
            )
    statement_events = [
        event for event in facts["sound_events"] if event.get("statement_id")
    ]
    if statement_events:
        statement_slots = {event["source_slot_id"] for event in statement_events}
        _require(
            len(statement_slots) == 1,
            "appearance-to-content binder needs one spoken-content source slot",
        )
        statement_slot = next(iter(statement_slots))
        instance = next(
            item for item in facts["instances"] if item["source_slot_id"] == statement_slot
        )
        candidate_fields = [
            ("sex_or_gender_label", instance["attributes"].get("sex_or_gender_label")),
            ("breed_id", instance.get("breed_id")),
            ("coat_value", instance["attributes"].get("coat_value")),
            ("body_build", instance["attributes"].get("body_build")),
            ("size", instance["attributes"].get("size")),
            ("life_stage", instance["attributes"].get("life_stage")),
        ]
        appearance_field = None
        appearance_value = None
        for field, value in candidate_fields:
            if value is None:
                continue
            matches = [
                item
                for item in facts["instances"]
                if (
                    item.get("breed_id")
                    if field == "breed_id"
                    else item["attributes"].get(field)
                )
                == value
            ]
            if len(matches) == 1:
                appearance_field, appearance_value = field, value
                break
        _require(
            appearance_field is not None,
            "spoken-content source has no unique controlled appearance field",
        )
        specs.append(
            {
                "schema": "avengine_qa_question_spec_v1",
                "spec_id": "QS-012",
                "question_type": "appearance_to_spoken_content",
                "selectors": {
                    "appearance_field": appearance_field,
                    "appearance_value": appearance_value,
                },
            }
        )
    return (
        specs,
        {
            "schema": "avengine_qa_controlled_sound_registry_v1",
            "registry_id": f"{facts['episode_id']}_controlled_sounds_v1",
            "sound_assets": sound_assets,
        },
        bindings,
    )


def build(
    *,
    manifest_path: Path,
    fact_path: Path,
    asset_registry_path: Path,
    output: Path,
    occluder_evidence_path: Path | None = None,
) -> Mapping[str, Any]:
    source_facts = _load(fact_path)
    _require(source_facts.get("status") == "pass", "source Facts did not pass")
    declared_asset_registries = [
        item
        for item in source_facts["provenance"]["inputs"]
        if item.get("role") == "source_asset_runtime_registry"
    ]
    _require(
        len(declared_asset_registries) == 1
        and _sha256(asset_registry_path)
        == declared_asset_registries[0]["sha256"],
        "QuestionSpec asset registry must match the registry declared by source Facts",
    )
    manifest, pixel_truth, artifact_paths = _validate_capture(
        manifest_path=manifest_path, facts=source_facts
    )
    request = manifest["authoritative_capture_request"]
    scenario_type = request["scenario_type"]
    target_slot = request["target_source_slot_id"]
    facts = deepcopy(source_facts)
    facts["visibility"]["pixel_truth"] = pixel_truth
    if occluder_evidence_path is not None:
        occluder_evidence = _load(occluder_evidence_path)
        _require(
            occluder_evidence.get("camera_pose_ids")
            == manifest["frame_contract"]["camera_pose_ids"],
            "occluder evidence camera poses differ from native capture",
        )
        facts["visibility"]["occluder_evidence"] = occluder_evidence
    facts["provenance"]["inputs"].append(
        {
            "role": "native_spear_pixel_capture_manifest",
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
        }
    )
    schema = _load(REPOSITORY / "schemas/avengine_qa_fact_table_v1.schema.json")
    jsonschema.validate(facts, schema)
    specs, sound_registry, bindings = _question_inputs(
        facts, scenario_type=scenario_type, target_slot=target_slot
    )
    evaluations = evaluate_question_specs(
        specs,
        facts=facts,
        asset_registry=_load(asset_registry_path),
        sound_registry=sound_registry,
        event_sound_bindings=bindings,
    )
    by_id = {item["spec_id"]: item for item in evaluations}
    scenario_observation: dict[str, Any]
    if scenario_type == "offscreen_to_onscreen":
        target_frames = pixel_truth["per_instance"][target_slot]["frames"]
        visible_after_frame_zero = [
            frame["frame_index"]
            for frame in target_frames[1:]
            if frame["visible_pixels"] > 0
        ]
        _require(
            target_frames[0]["target_pixels"] == 0
            and target_frames[0]["visible_pixels"] == 0
            and target_frames[0]["state"] == "out_of_view"
            and visible_after_frame_zero,
            "entry scenario requires frame0 target=visible=0 and a later visible target",
        )
        _require(
            by_id["QS-007"]["status"] == "pass"
            and by_id["QS-007"]["answer"]["value"] in {"left", "right"},
            "offscreen_to_onscreen required QS-007 did not produce a unique entry side "
            f"answer: {by_id['QS-007']}",
        )
        scenario_observation = {
            "observed_scenario_type": "offscreen_to_onscreen",
            "frame0_target_pixels": target_frames[0]["target_pixels"],
            "frame0_visible_pixels": target_frames[0]["visible_pixels"],
            "first_visible_frame": visible_after_frame_zero[0],
        }
    elif scenario_type == "occlusion_to_reappearance":
        target_frames = pixel_truth["per_instance"][target_slot]["frames"]
        occluded_frames = [
            frame["frame_index"]
            for frame in target_frames
            if frame["state"] == "visible_occluded"
        ]
        clear_after_occlusion = [
            frame["frame_index"]
            for frame in target_frames
            if frame["state"] == "visible_clear"
            and any(index < frame["frame_index"] for index in occluded_frames)
        ]
        _require(
            occluded_frames and clear_after_occlusion,
            "occlusion scenario requires native partial occlusion followed by a clear frame",
        )
        _require(
            by_id["QS-008"]["status"] == "pass"
            and by_id["QS-008"]["answer"]["value"] == "visible_occluded",
            "occlusion scenario requires QS-008=visible_occluded at the anchor frame",
        )
        _require(
            by_id["QS-009"]["status"] == "pass"
            and by_id["QS-009"]["answer"]["value"] == "no",
            "partial-occlusion scenario must honestly reject full-occlusion reappearance",
        )
        _require(
            by_id["QS-011"]["status"] == "pass"
            and by_id["QS-011"]["answer"]["value"] == "yes",
            "occlusion scenario requires QS-011 partial-occlusion-to-clear=yes",
        )
        _require(
            by_id["QS-010"]["status"] == "pass",
            "occlusion scenario requires a unique native static occluder answer",
        )
        scenario_observation = {
            "observed_scenario_type": "partial_occlusion_to_clear",
            "occluded_frame_indices": occluded_frames,
            "first_clear_frame_after_occlusion": clear_after_occlusion[0],
            "occlusion_anchor_frame": OCCLUSION_ANCHOR_FRAME,
            "occluder_answer": by_id["QS-010"]["answer"],
        }
    elif scenario_type == "full_occlusion_to_reappearance":
        target_frames = pixel_truth["per_instance"][target_slot]["frames"]
        fully_occluded_frames = [
            frame["frame_index"]
            for frame in target_frames
            if frame["state"] == "fully_occluded"
        ]
        reappeared_frames = by_id["QS-009"]["evidence"].get(
            "reappeared_frames", []
        )
        _require(
            fully_occluded_frames and reappeared_frames,
            "full-occlusion scenario requires native fully_occluded then visible frames",
        )
        _require(
            by_id["QS-008"]["status"] == "pass"
            and by_id["QS-008"]["answer"]["value"] == "fully_occluded",
            "full-occlusion scenario requires QS-008=fully_occluded",
        )
        _require(
            by_id["QS-009"]["status"] == "pass"
            and by_id["QS-009"]["answer"]["value"] == "yes",
            "full-occlusion scenario requires QS-009=yes",
        )
        if "QS-010" in by_id:
            _require(
                by_id["QS-010"]["status"] == "pass",
                "emitted full-occlusion identity question must have a unique answer",
            )
        if "QS-012" in by_id:
            _require(
                by_id["QS-012"]["status"] == "pass",
                "appearance-to-spoken-content QuestionSpec did not pass",
            )
        scenario_observation = {
            "observed_scenario_type": "full_occlusion_to_reappearance",
            "fully_occluded_frame_indices": fully_occluded_frames,
            "reappeared_frame_indices": reappeared_frames,
            "occlusion_anchor_frame": by_id["QS-008"]["evidence"]["frame_index"],
            "occluder_answer": (
                by_id["QS-010"]["answer"] if "QS-010" in by_id else None
            ),
            "occluder_question_status": (
                "emitted_unique_answer"
                if "QS-010" in by_id
                else "omitted_no_unique_single_object_answer"
            ),
            "appearance_to_content_answer": (
                by_id["QS-012"]["answer"] if "QS-012" in by_id else None
            ),
        }
    else:
        raise RuntimeError(f"unsupported native scenario type: {scenario_type}")

    output.mkdir(parents=True)
    output_files = {
        "facts": output / "facts.json",
        "question_specs": output / "question_specs.json",
        "question_evaluations": output / "question_evaluations.json",
        "sound_registry": output / "controlled_sound_registry.json",
        "event_sound_bindings": output / "event_sound_bindings.json",
    }
    write_json(output_files["facts"], facts)
    write_json(output_files["question_specs"], specs)
    write_json(output_files["question_evaluations"], evaluations)
    write_json(output_files["sound_registry"], sound_registry)
    write_json(output_files["event_sound_bindings"], bindings)
    binding = {
        "schema": SCHEMA,
        "status": "pass",
        "episode_id": facts["episode_id"],
        "scenario_type": scenario_type,
        "target_instance_id": target_slot,
        "pixel_authority": pixel_truth["authority"],
        "pixel_state_counts": pixel_truth["per_instance"][target_slot][
            "state_counts"
        ],
        "question_status_by_spec": {
            item["spec_id"]: item["status"] for item in evaluations
        },
        "question_answer_by_spec": {
            item["spec_id"]: item.get("answer") for item in evaluations
        },
        "scenario_observation": scenario_observation,
        "capture_manifest": _file_record(manifest_path),
        "source_fact": _file_record(fact_path),
        "source_asset_registry": _file_record(asset_registry_path),
        "occluder_evidence": (
            None
            if occluder_evidence_path is None
            else _file_record(occluder_evidence_path)
        ),
        "native_artifacts": {
            name: manifest["artifact_records"][name]
            for name in (
                "native_rgb_binaural",
                "metric_depth",
                "pixel_masks",
                "pixel_visibility_truth",
                "runtime_readbacks",
                "normal_object_ids",
                "object_id_descriptors",
                "rgb_frames",
            )
        },
        "outputs": {
            name: _file_record(path) for name, path in output_files.items()
        },
        "contract_sha256": canonical_json_sha256(
            {
                "episode_id": facts["episode_id"],
                "capture_manifest_sha256": _sha256(manifest_path),
                "facts_sha256": _sha256(output_files["facts"]),
                "question_evaluations_sha256": _sha256(
                    output_files["question_evaluations"]
                ),
            }
        ),
    }
    write_json(output / "manifest.json", binding)
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--fact", required=True, type=Path)
    parser.add_argument(
        "--asset-registry",
        type=Path,
        default=REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--occluder-evidence", type=Path)
    args = parser.parse_args()
    result = build(
        manifest_path=args.capture_manifest.resolve(),
        fact_path=args.fact.resolve(),
        asset_registry_path=args.asset_registry.resolve(),
        output=args.output.resolve(),
        occluder_evidence_path=(
            None
            if args.occluder_evidence is None
            else args.occluder_evidence.resolve()
        ),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
