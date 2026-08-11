from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path

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


def _module():
    spec = importlib.util.spec_from_file_location(
        "strict2h_global100_builder", BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _module()


class FrozenGlobalAssignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
        self.assignment = json.loads(ASSIGNMENT_PATH.read_text(encoding="utf-8"))

    def test_exact_global_and_per_batch_balances_without_cross_quota(self) -> None:
        rows = BUILDER._validate_frozen_assignment_structure(
            self.request, self.assignment
        )

        self.assertEqual(len(rows), 100)
        self.assertEqual(
            len({row["native_source_scenario_id"] for row in rows}), 100
        )
        self.assertEqual(len({row["camera_cluster_id"] for row in rows}), 100)
        self.assertEqual(
            Counter(row["mechanism"] for row in rows),
            Counter(
                {
                    mechanism: 20
                    for mechanism in self.request["mechanism_schedule"]
                }
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
        expected_strata = Counter(
            {f"stratum_{index:02d}": 2 for index in range(1, 6)}
        )
        for batch_number in range(1, 11):
            batch = [
                row
                for row in rows
                if row["batch_id"] == f"batch_{batch_number:02d}"
            ]
            self.assertEqual(
                Counter(
                    (row["mechanism"], row["target_side"]) for row in batch
                ),
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
            budget["rir_cache_empirical_budget"][
                "empirical_extrapolation_decimal_gb"
            ],
            1.2598373357894739,
        )
        self.assertEqual(
            budget["rir_cache_empirical_budget"]["budget_decimal_gb"], 1.3
        )
        self.assertEqual(
            budget["minimum_capture_plus_rir_workspace_decimal_gb"], 12.8
        )
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

            self.assertEqual(
                validation["status"],
                "pass_exact_cpu_assignment_not_native_execution",
            )
            self.assertEqual(
                validation["assignment_mode"],
                "frozen_global_assignment_pure_stdlib_revalidation",
            )
            self.assertFalse(validation["solver_required_at_builder_runtime"])
            self.assertFalse(
                validation["fixed_mechanism_stratum_cross_quota_required"]
            )
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
            self.assertEqual(
                summary["capture_media_only_estimated_storage_gb"], 11.5
            )
            self.assertEqual(
                summary["empirical_rir_cache_estimated_storage_gb"], 1.3
            )
            self.assertEqual(summary["minimum_workspace_storage_gb"], 12.8)
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
                    episode["dynamic_rir_state_budget"][
                        "requested_source_frame_uses"
                    ]
                    == 150
                    for episode in manifest["episodes"]
                )
            )
            self.assertEqual(
                len(list((Path(directory) / "global100/batches").glob("batch_*.json"))),
                10,
            )


if __name__ == "__main__":
    unittest.main()
