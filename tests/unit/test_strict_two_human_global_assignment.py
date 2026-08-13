from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator
from avengine.qa.actor_motion_profile import (
    build_actor_motion_profile_from_planning,
    validate_actor_motion_profile,
)


REPOSITORY = Path(__file__).resolve().parents[2]
BUILDER_PATH = REPOSITORY / "tools/qa/build_strict_two_human_full_episode_batch.py"
REQUEST_PATH = (
    REPOSITORY / "examples/qa/native_strict_two_human_full_episode_batch_v1.json"
)
ASSIGNMENT_PATH = (
    REPOSITORY
    / "examples/qa/native_strict_two_human_full_episode_global_assignment_v1.json"
)
NATIVE_SUITE = Path(
    "/data/datasets/avengine_workspaces/AVEngine-habitat-native/tmp/m7/"
    "apartment_asset_bound_ue_unique1000_full_20260723_01/"
    "suite_execution_plan.json"
)
RUNTIME_REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"


def _module():
    spec = importlib.util.spec_from_file_location(
        "strict2h_global100_builder", BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _module()


def _bind_runtime_import_fixture(tmp_path: Path, record: dict) -> None:
    spear = record["runtime_backends"]["spear_unreal"]
    ref = spear["ue_import_manifest_ref"]
    tag = ref["tag"]
    runtime_root = (
        "rocketbox_native_runtime_ue_v3"
        if ref["schema"] == "rocketbox_native_ue_import_v3"
        else "rocketbox_batch_native_runtime_ue_v1"
    )
    source_glb = tmp_path / runtime_root / tag / "runtime.glb"
    source_glb.parent.mkdir(parents=True, exist_ok=True)
    source_glb.write_bytes(b"fixture")
    manifest_path = tmp_path / ref["schema"] / tag / "ue_import_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_directory = spear["idle_animation"].rsplit("/", 1)[0]
    blueprint = spear["blueprint_class_path"].split(".", 1)[0]
    manifest = {
        "schema": ref["schema"],
        "tag": tag,
        "asset_id": ref["import_asset_id"],
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "source_glb": str(source_glb),
        "reload_verification": {"status": "passed"},
        "runtime_contract": {
            "actor_scale": spear["actor_scale"],
            "bone_count": 80,
            "bounds": {"height_passed": True, "ground_passed": True},
        },
        "glb_contract": {
            "armature_scale": [1.0, 1.0, 1.0],
            "armature_translation": [0.0, 0.0, 0.0],
            "animation_names": ["Standing_Idle", "Walking"],
            "joint_count": 80,
            "skin_count": 1,
            "mesh_count": 1,
            "mesh_is_scene_root": True,
        },
        "content": {
            "blueprint": blueprint,
            "skeletal_mesh": f"{mesh_directory}/runtime.runtime",
            "skeleton": f"{mesh_directory}/runtime_Skeleton.runtime_Skeleton",
            "animations": {
                "Standing_Idle": spear["idle_animation"],
                "Walking": spear["walking_animation"],
            },
        },
    }
    if "base_avatar_id" in ref:
        manifest["base_avatar_id"] = ref["base_avatar_id"]
    manifest_path.write_text(json.dumps(manifest))
    ref["path"] = str(manifest_path)


def test_runtime_motion_projection_uses_concrete_asset_periods(tmp_path: Path) -> None:
    registry_path = RUNTIME_REGISTRY
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for record in registry["assets"]:
        if "ue_import_manifest_ref" in record.get("runtime_backends", {}).get(
            "spear_unreal", {}
        ):
            _bind_runtime_import_fixture(tmp_path, record)
    fixture_registry_path = tmp_path / "runtime_registry.json"
    fixture_registry_path.write_text(json.dumps(registry))
    profiles = BUILDER._runtime_motion_profiles(fixture_registry_path, registry)

    female = profiles[
        (
            "lead_b_rocketbox_adults_female_adult_01_original_v1",
            "native_runtime_ue_v1",
        )
    ]
    male = profiles[
        (
            "rocketbox_human_male_adult_01_m5_1_candidate",
            "native_runtime_ue_v3",
        )
    ]
    assert female["walk_phase_period_frames"] == 19
    assert male["walk_phase_period_frames"] == 16
    assert female["animation_paths_by_action_id"]["walk"].endswith("/Walking.Walking")
    assert female["runtime_registry"] == str(fixture_registry_path.resolve())
    construction = profiles[
        (
            "lead_b_rocketbox_professions_construction_male_01_original_v1",
            "native_runtime_ue_v1",
        )
    ]
    assert construction["walk_phase_period_frames"] == 16
    assert construction["emitter_offset_m"] == [0.0, 1.664033031463623, 0.0]


def test_runtime_registry_import_ref_schema_is_exact() -> None:
    registry = json.loads(RUNTIME_REGISTRY.read_text(encoding="utf-8"))
    schema_path = REPOSITORY / "schemas/source_asset_runtime_registry_v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors(registry))
    record = next(
        item
        for item in registry["assets"]
        if item["asset_id"] == "rocketbox_human_male_adult_01_m5_1_candidate"
    )
    relative = deepcopy(registry)
    relative_record = next(
        item for item in relative["assets"] if item["asset_id"] == record["asset_id"]
    )
    relative_record["runtime_backends"]["spear_unreal"]["ue_import_manifest_ref"][
        "path"
    ] = "relative/ue_import_manifest.json"
    assert list(validator.iter_errors(relative))
    extra = deepcopy(registry)
    extra_record = next(
        item for item in extra["assets"] if item["asset_id"] == record["asset_id"]
    )
    extra_record["runtime_backends"]["spear_unreal"]["ue_import_manifest_ref"][
        "file_sha256"
    ] = "forbidden"
    assert list(validator.iter_errors(extra))


class FrozenGlobalAssignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
        self.assignment = json.loads(ASSIGNMENT_PATH.read_text(encoding="utf-8"))

    def test_exact_global_and_per_batch_balances_without_cross_quota(self) -> None:
        rows = BUILDER._validate_frozen_assignment_structure(
            self.request, self.assignment
        )

        self.assertEqual(len(rows), 100)
        self.assertEqual(len({row["native_source_scenario_id"] for row in rows}), 100)
        self.assertEqual(len({row["camera_cluster_id"] for row in rows}), 100)
        self.assertEqual(
            Counter(row["mechanism"] for row in rows),
            Counter(
                {mechanism: 20 for mechanism in self.request["mechanism_schedule"]}
            ),
        )
        self.assertEqual(
            Counter(row["target_side"] for row in rows),
            Counter({"left": 50, "right": 50}),
        )
        self.assertEqual(
            Counter(row["stratum_id"] for row in rows),
            Counter({f"stratum_{index:02d}": 20 for index in range(1, 6)}),
        )
        expected_mechanism_side = Counter(
            {
                (mechanism, side): 1
                for mechanism in self.request["mechanism_schedule"]
                for side in ("left", "right")
            }
        )
        expected_strata = Counter({f"stratum_{index:02d}": 2 for index in range(1, 6)})
        for batch_number in range(1, 11):
            batch = [
                row for row in rows if row["batch_id"] == f"batch_{batch_number:02d}"
            ]
            self.assertEqual(
                Counter((row["mechanism"], row["target_side"]) for row in batch),
                expected_mechanism_side,
            )
            self.assertEqual(
                Counter(row["stratum_id"] for row in batch), expected_strata
            )

        mechanism_stratum = Counter(
            (row["mechanism"], row["stratum_id"]) for row in rows
        )
        self.assertGreater(len(set(mechanism_stratum.values())), 1)
        self.assertEqual(mechanism_stratum[("target_moves", "stratum_01")], 0)

    def test_structure_validation_fails_closed(self) -> None:
        broken = deepcopy(self.assignment)
        broken["rows"][0]["stratum_id"] = broken["rows"][1]["stratum_id"]
        with self.assertRaisesRegex(
            RuntimeError, "spatial balance drift|stratum balance drift"
        ):
            BUILDER._validate_frozen_assignment_structure(self.request, broken)

    def test_builder_has_no_scipy_runtime_import(self) -> None:
        source = BUILDER_PATH.read_text(encoding="utf-8").lower()
        self.assertNotIn("import scipy", source)
        self.assertNotIn("from scipy", source)

    def test_storage_budget_separates_capture_media_and_empirical_rir(self) -> None:
        budget = BUILDER._storage_budget_summary(
            resource_budget=self.request["resource_budget"],
            episode_count=100,
            exact_rir_state_count=9080,
        )

        self.assertEqual(budget["capture_media_only"]["budget_decimal_gb"], 11.5)
        self.assertAlmostEqual(
            budget["capture_media_only"]["empirical_extrapolation_decimal_gb"],
            11.4047991,
        )
        self.assertEqual(
            budget["rir_cache_empirical_budget"]["reference_state_count"], 76
        )
        self.assertEqual(
            budget["rir_cache_empirical_budget"]["planned_exact_state_count"],
            9080,
        )
        self.assertAlmostEqual(
            budget["rir_cache_empirical_budget"]["empirical_extrapolation_decimal_gb"],
            1.2598373357894739,
        )
        self.assertEqual(budget["rir_cache_empirical_budget"]["budget_decimal_gb"], 1.3)
        self.assertEqual(budget["minimum_capture_plus_rir_workspace_decimal_gb"], 12.8)
        self.assertNotEqual(budget["excluded_from_minimum"], [])

    @unittest.skipUnless(
        NATIVE_SUITE.is_file(),
        "retained native Apartment 1000-Episode authority is not mounted",
    )
    def test_real_assignment_recomputes_all_100_cpu_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = BUILDER.build(REQUEST_PATH, Path(directory) / "global100")
            validation = json.loads(
                paths["assignment_validation"].read_text(encoding="utf-8")
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            construction_row = next(
                episode
                for episode in manifest["episodes"]
                if "lead_b_rocketbox_professions_construction_male_01_original_v1"
                in {
                    episode["target"]["runtime_asset_id"],
                    episode["distractor"]["runtime_asset_id"],
                }
            )
            construction_manifest = Path(directory) / "construction_consumer.json"
            construction_manifest.write_text(
                json.dumps({"episodes": [construction_row]}),
                encoding="utf-8",
            )
            construction_profile = build_actor_motion_profile_from_planning(
                planning_manifest_path=construction_manifest,
                episode_id=construction_row["episode_id"],
            )
            validate_actor_motion_profile(construction_profile)
            construction_assets = {
                declaration["asset_id"]
                for declaration in construction_profile["authorities"]["candidate"][
                    "value"
                ]["actor_declarations"].values()
            }
            self.assertIn(
                "lead_b_rocketbox_professions_construction_male_01_original_v1",
                construction_assets,
            )

            self.assertEqual(
                validation["status"],
                "pass_exact_cpu_assignment_not_native_execution",
            )
            self.assertEqual(
                validation["assignment_mode"],
                "frozen_global_assignment_pure_stdlib_revalidation",
            )
            self.assertFalse(validation["solver_required_at_builder_runtime"])
            self.assertFalse(validation["fixed_mechanism_stratum_cross_quota_required"])
            self.assertEqual(validation["episode_count"], 100)
            self.assertEqual(validation["unique_source_scenario_count"], 100)
            self.assertEqual(validation["unique_camera_cluster_count"], 100)
            self.assertGreaterEqual(validation["minimum_moving_path_length_m"], 0.5)
            self.assertGreaterEqual(
                validation["minimum_actor_horizontal_separation_m"], 1.0
            )
            self.assertLessEqual(
                validation["maximum_native_source_root_readback_drift_m"], 1.0e-6
            )
            self.assertEqual(manifest["episode_count"], 100)
            self.assertEqual(len(manifest["episodes"]), 100)
            self.assertEqual(summary["exact_rir_state_count_required"], 9080)
            self.assertEqual(summary["requested_source_frame_uses"], 15000)
            self.assertNotIn("estimated_storage_gb", summary)
            self.assertEqual(summary["capture_media_only_estimated_storage_gb"], 11.5)
            self.assertEqual(summary["empirical_rir_cache_estimated_storage_gb"], 1.3)
            self.assertEqual(summary["minimum_workspace_storage_gb"], 12.8)
            self.assertTrue(
                summary["release_gate"][
                    "release_blocked_without_accepted_ground_contact_evidence"
                ]
            )
            self.assertEqual(summary["release_gate"], manifest["release_gate"])
            self.assertEqual(
                summary["exact_rir_state_count_by_mechanism"],
                {
                    "both_move": 3000,
                    "both_static": 40,
                    "camera_pan_both_static": 3000,
                    "distractor_moves": 1520,
                    "target_moves": 1520,
                },
            )
            self.assertTrue(
                all(
                    episode["dynamic_rir_state_budget"]["requested_source_frame_uses"]
                    == 150
                    for episode in manifest["episodes"]
                )
            )
            self.assertEqual(
                len(list((Path(directory) / "global100/batches").glob("batch_*.json"))),
                10,
            )
            first_batch = json.loads(
                (Path(directory) / "global100/batches/batch_01.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(first_batch["release_gate"], summary["release_gate"])

            row = next(
                item
                for item in manifest["episodes"]
                if item["episode_id"] == "strict2h_full75_0002_v1"
            )
            target = row["target"]["motion_profile_authority"]
            distractor = row["distractor"]["motion_profile_authority"]
            self.assertEqual(target["source_path"]["source_suite"], str(NATIVE_SUITE))
            self.assertEqual(
                target["source_path"]["native_source_scenario_id"],
                "cat_border_collie__recombined_source1_static_source2_moving_0265",
            )
            self.assertEqual(target["source_path"]["source_actor_id"], "source2_actor")
            self.assertEqual(target["source_path"]["frame_index_map"], list(range(75)))
            self.assertEqual(
                target["runtime"]["asset_id"],
                "lead_b_rocketbox_adults_female_adult_01_original_v1",
            )
            self.assertEqual(target["runtime"]["walk_phase_period_frames"], 19)
            self.assertEqual(
                distractor["source_path"]["source_actor_id"], "source1_actor"
            )
            self.assertEqual(distractor["source_path"]["frame_index_map"], [0] * 75)
            self.assertEqual(distractor["runtime"]["walk_phase_period_frames"], 16)
            self.assertEqual(
                manifest["runtime_registry"], str(RUNTIME_REGISTRY.resolve())
            )


if __name__ == "__main__":
    unittest.main()
