from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


WRAPPER_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/qa/capture_skokloster_strict_two_human_episode.py"
)
SCENARIO_ID = "skokloster_diagnostic_case"


def _load_wrapper() -> ModuleType:
    name = "avengine_test_skokloster_capture_role"
    spec = importlib.util.spec_from_file_location(name, WRAPPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WRAPPER = _load_wrapper()


def _suite() -> dict[str, object]:
    return {
        "backend_role": "comparison_visual",
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "native_map": WRAPPER.PACKAGED_MAP,
        "packaged_executable": (
            "/external/Standalone-Skokloster-Development/Linux/SpearSim.sh"
        ),
        "scenarios": [
            {
                # Canonical Skokloster scenarios carry no direct claim fields.
                "scenario_id": SCENARIO_ID,
                "backend_role": "comparison_visual",
                "plan": {
                    "backend_role": "comparison_visual",
                    "qualification": {
                        "qualification_claim": False,
                        "formal_dataset_count": 0,
                    },
                },
            }
        ],
    }


def _role_container(suite: dict[str, object], level: str) -> dict[str, object]:
    if level == "suite":
        return suite
    scenario = suite["scenarios"][0]
    assert isinstance(scenario, dict)
    if level == "scenario":
        return scenario
    plan = scenario["plan"]
    assert isinstance(plan, dict)
    return plan


def _claim_container(suite: dict[str, object], level: str) -> dict[str, object]:
    if level == "suite":
        return suite
    scenario = suite["scenarios"][0]
    assert isinstance(scenario, dict)
    if level == "scenario":
        return scenario
    plan = scenario["plan"]
    assert isinstance(plan, dict)
    qualification = plan["qualification"]
    assert isinstance(qualification, dict)
    return qualification


def test_accepts_normal_canonical_plan_without_scenario_claim_fields() -> None:
    WRAPPER.validate_diagnostic_suite(_suite(), scenario_id=SCENARIO_ID)


def test_does_not_require_plan_qualification_object() -> None:
    suite = _suite()
    plan = _role_container(suite, "plan")
    plan.pop("qualification")

    WRAPPER.validate_diagnostic_suite(suite, scenario_id=SCENARIO_ID)


def test_run_rejects_production_before_runtime_configuration(tmp_path: Path) -> None:
    suite = _suite()
    suite["backend_role"] = "production_visual"
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")

    args = SimpleNamespace(
        authorize_gpu_capture=True,
        output=tmp_path / "new-output",
        suite_plan=suite_path,
        scenario_id=SCENARIO_ID,
    )
    with pytest.raises(RuntimeError, match="Skokloster suite backend_role"):
        WRAPPER.run(args)


@pytest.mark.parametrize("level", ("suite", "scenario", "plan"))
def test_rejects_production_role_at_each_scope(level: str) -> None:
    suite = deepcopy(_suite())
    _role_container(suite, level)["backend_role"] = "production_visual"

    with pytest.raises(RuntimeError, match=rf"Skokloster {level} backend_role"):
        WRAPPER.validate_diagnostic_suite(suite, scenario_id=SCENARIO_ID)


@pytest.mark.parametrize("level", ("suite", "scenario", "plan"))
def test_rejects_qualification_claim_at_each_scope(level: str) -> None:
    suite = deepcopy(_suite())
    _claim_container(suite, level)["qualification_claim"] = True
    owner = "plan qualification" if level == "plan" else level

    with pytest.raises(RuntimeError, match=rf"Skokloster {owner} qualification_claim"):
        WRAPPER.validate_diagnostic_suite(suite, scenario_id=SCENARIO_ID)


@pytest.mark.parametrize("level", ("suite", "scenario", "plan"))
def test_rejects_nonzero_formal_count_at_each_scope(level: str) -> None:
    suite = deepcopy(_suite())
    _claim_container(suite, level)["formal_dataset_count"] = 1
    owner = "plan qualification" if level == "plan" else level

    with pytest.raises(RuntimeError, match=rf"Skokloster {owner} formal_dataset_count"):
        WRAPPER.validate_diagnostic_suite(suite, scenario_id=SCENARIO_ID)


def test_uses_shared_runner_last_entry_selection_without_uniqueness_requirement() -> None:
    suite = _suite()
    scenarios = suite["scenarios"]
    assert isinstance(scenarios, list)
    shadowed = deepcopy(scenarios[0])
    shadowed["backend_role"] = "production_visual"
    scenarios.insert(0, shadowed)

    WRAPPER.validate_diagnostic_suite(suite, scenario_id=SCENARIO_ID)


def test_ignores_unselected_scenario_claims() -> None:
    suite = _suite()
    scenarios = suite["scenarios"]
    assert isinstance(scenarios, list)
    scenarios.append(
        {
            "scenario_id": "unselected-production-scenario",
            "backend_role": "production_visual",
            "qualification_claim": True,
            "formal_dataset_count": 1,
            "plan": {"backend_role": "production_visual"},
        }
    )

    WRAPPER.validate_diagnostic_suite(suite, scenario_id=SCENARIO_ID)
