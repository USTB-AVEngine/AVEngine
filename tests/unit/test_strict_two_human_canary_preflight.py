from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/build_strict_two_human_canary_preflight.py"
TOOL_SPEC = importlib.util.spec_from_file_location(
    "build_strict_two_human_canary_preflight", TOOL_PATH
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
TOOL = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(TOOL)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict]:
    plan = _load(REPOSITORY / "examples/qa/native_strict_two_human_canary_v1.json")
    report = _load(REPOSITORY / "reports/lead_a/two_human_feasibility_v1.json")
    registry = _load(REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json")
    return plan, report, registry


def test_strict_two_human_canary_cpu_contract_passes() -> None:
    plan, report, registry = _inputs()

    assert TOOL.validate_contract(plan, report, registry) == []
    female = next(
        asset
        for asset in registry["assets"]
        if asset["asset_id"]
        == "lead_b_rocketbox_adults_female_adult_01_original_v1"
    )
    assert female["revision"] == "native_runtime_ue_v1"
    assert female["identity"] == {"species_id": "human", "breed_id": None}
    assert female["realized_attributes"] == {
        "life_stage": "adult",
        "sex_or_gender_label": "female",
    }
    assert female["emitter_anchors"][0]["offset_m"] == [
        0.0,
        1.569012451171875,
        0.0,
    ]
    unreal = female["runtime_backends"]["spear_unreal"]
    assert "gate_rocketbox_adults_female_adult_01_original_ue_v1" in unreal[
        "blueprint_class_path"
    ]
    assert unreal["idle_animation"].endswith("/Standing_Idle.Standing_Idle")
    assert unreal["walking_animation"].endswith("/Walking.Walking")


def test_strict_two_human_canary_rejects_same_identity_or_material_variant() -> None:
    plan, report, registry = _inputs()
    invalid = deepcopy(plan)
    invalid["actors"][1]["original_identity_id"] = invalid["actors"][0][
        "original_identity_id"
    ]
    errors = TOOL.validate_contract(invalid, report, registry)
    assert "distractor original identity mismatch" in errors
    assert "actors must use distinct original identities" in errors

    invalid = deepcopy(plan)
    female_id = invalid["actors"][1]["runtime_asset_id"]
    male_profile = next(
        asset for asset in registry["assets"] if asset["asset_id"] == invalid["actors"][0]["runtime_asset_id"]
    )
    variant_registry = deepcopy(registry)
    female_profile = next(
        asset for asset in variant_registry["assets"] if asset["asset_id"] == female_id
    )
    female_profile["geometry"]["source_mesh_uri"] = male_profile["geometry"][
        "source_mesh_uri"
    ]
    errors = TOOL.validate_contract(invalid, report, variant_registry)
    assert "material variants cannot establish distinct identity" in errors


def test_strict_two_human_canary_rejects_voice_gpu_or_catalog_drift() -> None:
    plan, report, registry = _inputs()

    invalid = deepcopy(plan)
    invalid["actors"][1]["voice_policy"] = "speaking"
    invalid["actors"][1]["sound_events"] = [{"event_id": "forbidden"}]
    errors = TOOL.validate_contract(invalid, report, registry)
    assert "distractor must be silent" in errors
    assert "silent distractor cannot have sound events" in errors

    invalid = deepcopy(plan)
    invalid["gpu_policy"]["physical_gpu_index"] = 0
    invalid["paper_catalog_mutation_allowed"] = True
    errors = TOOL.validate_contract(invalid, report, registry)
    assert "physical GPU must be 1" in errors
    assert "paper catalog mutation is forbidden" in errors

    invalid = deepcopy(plan)
    invalid["rir_policy"]["reuse_existing_cache_as_exact_two_human_evidence"] = True
    errors = TOOL.validate_contract(invalid, report, registry)
    assert "existing RIR cache cannot be reused as exact two-human evidence" in errors

    invalid = deepcopy(plan)
    invalid["runtime_lineage"]["target"]["build_tag"] = "unrelated_build"
    errors = TOOL.validate_contract(invalid, report, registry)
    assert "target positive runtime lineage mismatch" in errors
