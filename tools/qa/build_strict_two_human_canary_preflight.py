#!/usr/bin/env python3
"""Validate and publish the CPU preflight for one strict two-human canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "avengine_native_strict_two_human_canary_plan_v1"
OUTPUT_SCHEMA = "avengine_native_strict_two_human_canary_preflight_v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repository_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPOSITORY / path


def _asset_by_id(registry: Mapping[str, Any], asset_id: str) -> Mapping[str, Any]:
    matches = [
        asset
        for asset in registry.get("assets", [])
        if asset.get("asset_id") == asset_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"runtime asset does not resolve exactly once: {asset_id}")
    return matches[0]


def validate_contract(
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> list[str]:
    """Return deterministic contract errors without touching external evidence."""

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(plan.get("schema") == PLAN_SCHEMA, "plan schema mismatch")
    require(
        plan.get("status") == "cpu_preflight_pending",
        "plan must remain pending until preflight is published",
    )
    require(report.get("status") == "ready_for_canary_cpu_registration", "report status mismatch")
    require(
        report.get("current_blocker")
        == "register_second_adult_in_a_runtime_profile",
        "feasibility blocker changed",
    )
    require(plan.get("paper_catalog_mutation_allowed") is False, "paper catalog mutation is forbidden")

    actors = plan.get("actors")
    require(isinstance(actors, list) and len(actors) == 2, "exactly two actors are required")
    if not isinstance(actors, list) or len(actors) != 2:
        return errors
    by_role = {actor.get("role"): actor for actor in actors}
    require(set(by_role) == {"target", "distractor"}, "target/distractor roles are required")
    if set(by_role) != {"target", "distractor"}:
        return errors
    target = by_role["target"]
    distractor = by_role["distractor"]

    require(target.get("source_slot_id") == "source1", "target must use source1")
    require(distractor.get("source_slot_id") == "source2", "distractor must use source2")
    require(
        target.get("original_identity_id") == "rocketbox_adults_male_adult_01",
        "target original identity mismatch",
    )
    require(
        distractor.get("original_identity_id")
        == "rocketbox_adults_female_adult_01",
        "distractor original identity mismatch",
    )
    require(
        target.get("original_identity_id") != distractor.get("original_identity_id"),
        "actors must use distinct original identities",
    )
    require(target.get("expected_screen_side") == "right", "target side must be right")
    require(
        distractor.get("expected_screen_side") == "left",
        "distractor side must be left",
    )
    require(target.get("voice_policy") == "speaking", "target must speak")
    require(distractor.get("voice_policy") == "silent", "distractor must be silent")
    require(distractor.get("sound_events") == [], "silent distractor cannot have sound events")
    require(
        target.get("sound_asset_id") == "speech_cremad_1001_ieo_neu_v1",
        "target controlled speech mismatch",
    )
    require(target.get("action_id") == "idle", "target canary action must be idle")
    require(distractor.get("action_id") == "idle", "distractor canary action must be idle")
    require(
        target.get("runtime_asset_id") != distractor.get("runtime_asset_id"),
        "actors must use distinct runtime assets",
    )

    timeline = plan.get("timeline", {})
    require(timeline.get("frame_count") == 75, "canary must contain 75 formal frames")
    require(timeline.get("frame_rate_hz") == 15, "canary frame rate must be 15 Hz")
    require(timeline.get("sparse_gate_frame_index") == 15, "sparse gate must use frame 15")
    speech_window = target.get("speech_frame_window", [])
    require(
        isinstance(speech_window, list)
        and len(speech_window) == 2
        and speech_window[0] <= 15 <= speech_window[1],
        "sparse gate frame must fall inside target speech window",
    )

    gpu = plan.get("gpu_policy", {})
    require(gpu.get("physical_gpu_index") == 1, "physical GPU must be 1")
    require(gpu.get("graphics_adapter_argument") == 1, "graphics adapter must be 1")
    require(gpu.get("required_idle_compute_process_count") == 0, "GPU must be idle")
    require(gpu.get("forbidden_physical_gpu_indices") == [0, 3], "GPU 0/3 must be forbidden")
    require(
        plan.get("target_only_actor_map")
        == {
            "source1": "rocketbox_adults_male_adult_01_actor",
            "source2": "rocketbox_adults_female_adult_01_actor",
        },
        "target-only actor map mismatch",
    )

    approved = set(report.get("approved_adults", []))
    require(target.get("original_identity_id") in approved, "target identity is not approved")
    require(
        distractor.get("original_identity_id") in approved,
        "distractor identity is not approved",
    )
    excluded = {
        entry.get("identity_id") for entry in report.get("excluded_adults", [])
    }
    require(
        "rocketbox_professions_medical_female_01" in excluded,
        "medical female evidence contradiction is not frozen",
    )
    children = report.get("children", {})
    require(children.get("voice_policy") == "silent_only", "children must remain silent-only")
    require(children.get("allowed_as_speaking_target") is False, "children cannot be speaking targets")

    aliases = registry.get("aliases", {})
    for actor in actors:
        alias = aliases.get(actor.get("runtime_asset_alias"), {})
        require(alias.get("asset_id") == actor.get("runtime_asset_id"), f"{actor['role']} alias asset mismatch")
        require(alias.get("revision") == actor.get("runtime_revision"), f"{actor['role']} alias revision mismatch")
        try:
            profile = _asset_by_id(registry, str(actor.get("runtime_asset_id")))
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        require(profile.get("revision") == actor.get("runtime_revision"), f"{actor['role']} profile revision mismatch")
        require(profile.get("entity_class") == "articulated_human", f"{actor['role']} is not articulated_human")
        require(profile.get("identity", {}).get("species_id") == "human", f"{actor['role']} species is not human")
        require(profile.get("realized_attributes", {}).get("life_stage") == "adult", f"{actor['role']} is not adult")
        require(profile.get("default_emitter_anchor_id") == "mouth", f"{actor['role']} mouth anchor is missing")
        require(profile.get("admission_state") == "research", f"{actor['role']} must remain research")

    try:
        target_profile = _asset_by_id(registry, str(target.get("runtime_asset_id")))
        distractor_profile = _asset_by_id(registry, str(distractor.get("runtime_asset_id")))
        require(
            target_profile.get("geometry", {}).get("source_mesh_uri")
            != distractor_profile.get("geometry", {}).get("source_mesh_uri"),
            "material variants cannot establish distinct identity",
        )
        target_ue = target_profile.get("runtime_backends", {}).get("spear_unreal", {})
        distractor_ue = distractor_profile.get("runtime_backends", {}).get("spear_unreal", {})
        require(
            target_ue.get("blueprint_class_path")
            != distractor_ue.get("blueprint_class_path"),
            "actors must use distinct UE blueprint classes",
        )
        require(
            distractor_profile.get("realized_attributes", {}).get("sex_or_gender_label")
            == "female",
            "distractor profile gender mismatch",
        )
        require(
            distractor_profile.get("emitter_anchors", [{}])[0].get("offset_m")
            == [0.0, 1.569012451171875, 0.0],
            "female mouth-height evidence changed",
        )
    except RuntimeError as exc:
        errors.append(str(exc))

    provenance = plan.get("runtime_provenance", {})
    require(
        provenance.get("same_runtime_build_revision_required") is False,
        "male v3 and female batch v1 must remain explicit distinct revisions",
    )
    require(
        provenance.get("distinct_original_base_identity_required") is True,
        "distinct base identity gate is required",
    )
    rir_policy = plan.get("rir_policy", {})
    require(
        rir_policy.get("existing_unique1000_cache_is_template_only") is True,
        "existing RIR cache must remain template-only",
    )
    require(
        rir_policy.get("reuse_existing_cache_as_exact_two_human_evidence") is False,
        "existing RIR cache cannot be reused as exact two-human evidence",
    )
    return errors


def _ue_blueprint_class(import_manifest: Mapping[str, Any]) -> str:
    blueprint = str(import_manifest.get("content", {}).get("blueprint", ""))
    leaf = blueprint.rsplit("/", 1)[-1]
    return f"{blueprint}.{leaf}_C"


def build(plan_path: Path, output: Path) -> Path:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    plan = _load_json(plan_path)
    report_path = _resolve_repository_path(str(plan.get("feasibility_report", "")))
    registry_path = _resolve_repository_path(str(plan.get("source_runtime_registry", "")))
    report = _load_json(report_path)
    registry = _load_json(registry_path)
    errors = validate_contract(plan, report, registry)
    if errors:
        raise RuntimeError("CPU contract failed: " + "; ".join(errors))

    actors = {actor["role"]: actor for actor in plan["actors"]}
    profiles = {
        role: _asset_by_id(registry, actor["runtime_asset_id"])
        for role, actor in actors.items()
    }
    evidence_paths = {
        key: Path(value) for key, value in plan.get("evidence", {}).items()
    }
    missing = [key for key, path in evidence_paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"authoritative evidence files are missing: {missing}")

    target_record = _load_json(evidence_paths["target_source_record"])
    distractor_record = _load_json(evidence_paths["distractor_source_record"])
    records = {"target": target_record, "distractor": distractor_record}
    expected_identities = {
        "target": "rocketbox_adults_male_adult_01",
        "distractor": "rocketbox_adults_female_adult_01",
    }
    expected_genders = {"target": "male", "distractor": "female"}
    for role, record in records.items():
        if record.get("base_avatar_id") != expected_identities[role]:
            raise RuntimeError(f"{role} B source identity mismatch")
        if record.get("gender") != expected_genders[role]:
            raise RuntimeError(f"{role} B gender mismatch")
        if record.get("life_stage") != "adult":
            raise RuntimeError(f"{role} B source is not adult")
        if record.get("state_classification") != "research_candidate":
            raise RuntimeError(f"{role} B source admission boundary changed")
        if record.get("formal_dataset_registration_authorized") is not False:
            raise RuntimeError(f"{role} unexpectedly became formal")
        qa = record.get("qa", {})
        for gate in (
            "runtime_and_UE_readback",
            "idle_silent_review",
            "walking_silent_review",
        ):
            if qa.get(gate) != "passed":
                raise RuntimeError(f"{role} B gate failed: {gate}")
    if (
        target_record.get("voice_policy", {}).get("sound_asset_id")
        != actors["target"]["sound_asset_id"]
    ):
        raise RuntimeError("target controlled speech does not match B candidate")

    source_fbx_paths = {
        role: record.get("source", {}).get("fbx", {}).get("path")
        for role, record in records.items()
    }
    if len(set(source_fbx_paths.values())) != 2 or None in source_fbx_paths.values():
        raise RuntimeError("source FBX paths do not prove two original identities")

    import_manifests = {
        "target": _load_json(evidence_paths["target_ue_import_manifest"]),
        "distractor": _load_json(evidence_paths["distractor_ue_import_manifest"]),
    }
    for role, manifest in import_manifests.items():
        if manifest.get("reload_verification", {}).get("status") != "passed":
            raise RuntimeError(f"{role} UE reload verification failed")
        unreal = profiles[role]["runtime_backends"]["spear_unreal"]
        if _ue_blueprint_class(manifest) != unreal["blueprint_class_path"]:
            raise RuntimeError(f"{role} UE blueprint evidence mismatch")
        animations = manifest.get("content", {}).get("animations", {})
        if animations.get("Standing_Idle") != unreal["idle_animation"]:
            raise RuntimeError(f"{role} Idle evidence mismatch")
        if animations.get("Walking") != unreal["walking_animation"]:
            raise RuntimeError(f"{role} Walking evidence mismatch")

    if target_record.get("runtime", {}).get("glb", {}).get("path") == str(
        evidence_paths["target_runtime_glb"]
    ):
        raise RuntimeError("male v3 must not be represented as B batch v1")
    if distractor_record.get("runtime", {}).get("glb", {}).get("path") != str(
        evidence_paths["distractor_runtime_glb"]
    ):
        raise RuntimeError("female batch v1 runtime evidence mismatch")

    receipt = _load_json(evidence_paths["rir_template_cache_receipt"])
    if receipt.get("status") != "pass" or receipt.get("full_plan_complete") is not True:
        raise RuntimeError("native RLR template cache is not complete")

    executable = Path(plan["environment"]["current_spear_executable"])
    if not executable.is_file():
        raise RuntimeError("current SPEAR executable is missing")
    resolved_executable = executable.resolve()
    packaged_manifest = resolved_executable.parent / "Manifest_UFSFiles_Linux.txt"
    if not packaged_manifest.is_file():
        raise RuntimeError("current packaged runtime manifest is missing")
    packaged_text = packaged_manifest.read_text(encoding="utf-8")
    packaged_tags = {
        "target": "gate_rocketbox_male_adult_01_original_ue_v3",
        "distractor": "gate_rocketbox_adults_female_adult_01_original_ue_v1",
    }
    for role, tag in packaged_tags.items():
        required = (
            f"Blueprints/{tag}/BP_{tag}.uasset",
            f"Meshes/{tag}/runtime.uasset",
            f"Meshes/{tag}/Standing_Idle.uasset",
            f"Meshes/{tag}/Walking.uasset",
        )
        missing_entries = [entry for entry in required if entry not in packaged_text]
        if missing_entries:
            raise RuntimeError(
                f"recook_required: {role} packaged runtime entries missing: {missing_entries}"
            )

    output.mkdir(parents=True)
    records_out = {
        key: {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for key, path in evidence_paths.items()
    }
    preflight = {
        "schema": OUTPUT_SCHEMA,
        "status": "pass",
        "claim_boundary": "CPU-only preflight; no GPU process was launched, no native two-human pixels exist, and the old unique1000 RIR cache is template-only rather than exact scene evidence.",
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": _sha256(plan_path),
        },
        "runtime_registry": {
            "path": str(registry_path.resolve()),
            "sha256": _sha256(registry_path),
            "revision": registry["revision"],
        },
        "actors": {
            role: {
                "original_identity_id": actors[role]["original_identity_id"],
                "runtime_asset_id": actors[role]["runtime_asset_id"],
                "runtime_revision": actors[role]["runtime_revision"],
                "voice_policy": actors[role]["voice_policy"],
            }
            for role in ("target", "distractor")
        },
        "runtime_provenance": plan["runtime_provenance"],
        "rir_policy": plan["rir_policy"],
        "source_fbx_paths": source_fbx_paths,
        "packaged_runtime": {
            "executable": str(resolved_executable),
            "manifest": str(packaged_manifest),
            "required_actor_tags": packaged_tags,
            "status": "pass",
        },
        "evidence_records": records_out,
        "cpu_gates": {
            "distinct_original_identities": "pass",
            "runtime_profiles": "pass",
            "ue_import_and_reload": "pass",
            "idle_and_walking": "pass",
            "mouth_anchor": "pass",
            "target_speech_candidate": "research_pending_listening_review",
            "distractor_silent": "pass",
            "packaged_runtime": "pass",
            "native_rlr_template": "pass",
            "exact_two_human_rir": "pending_required",
        },
        "native_sparse_gate": "blocked_on_exact_two_human_rir",
        "formal_scene_count": 0,
        "next_state": "ready_for_exact_two_human_rir_plan",
    }
    output_path = output / "preflight.json"
    output_path.write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPOSITORY / "examples/qa/native_strict_two_human_canary_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.plan.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
