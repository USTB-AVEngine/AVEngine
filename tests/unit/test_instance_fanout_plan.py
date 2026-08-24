"""The instance fan-out plan must be derived, breed-agnostic and fail-closed."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from avengine.appearance.contracts import (
    APPEARANCE_AXES,
    CANONICAL_DOMAINS,
    COAT_PROFILE_DOMAINS,
    OPERATION_BY_AXIS,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY_ROOT / "tools/assets/plan_instance_variants.py"
REGISTRY_PATH = REPOSITORY_ROOT / "examples/runtime/source_asset_runtime_profiles.json"
SPEC_PATH = REPOSITORY_ROOT / "examples/assets/instance_fanout_axes_v1.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("plan_instance_variants", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


@pytest.fixture(scope="module")
def payload():
    return TOOL.build_plan(REGISTRY_PATH, SPEC_PATH, None)


def test_shipped_spec_decides_every_contract_axis():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    decided = set(spec["enabled_axes"]) | set(spec["pinned_axes"])
    assert decided == set(APPEARANCE_AXES)
    assert set(spec["enabled_axes"]).isdisjoint(spec["pinned_axes"])


def test_plans_cover_every_appearance_realized_asset(payload):
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    eligible = {
        entry["asset_id"]
        for entry in registry["assets"]
        if isinstance((entry.get("realized_attributes") or {}).get("coat_profile"), dict)
    }
    planned = {plan["source_asset"]["asset_id"] for plan in payload["plans"]}
    skipped = {skip["asset_id"] for skip in payload["skipped"]}
    assert eligible <= planned | skipped
    assert planned, "no source asset produced a plan"


def test_each_plan_round_trips_its_own_registered_id(payload):
    """The derivation rule must reproduce the id the asset is already registered under."""
    for plan in payload["plans"]:
        source_id = plan["source_asset"]["asset_id"]
        identity_rows = [row for row in plan["rows"] if row["already_registered"]]
        assert len(identity_rows) == 1, source_id
        assert identity_rows[0]["asset_id"] == source_id
        assert not identity_rows[0]["requires_new_ue_asset"]


def test_row_count_is_the_enabled_axis_product(payload):
    for plan in payload["plans"]:
        expected = 1
        for axis in plan["enabled_axes"]:
            expected *= len(plan["axis_domains"][axis])
        assert plan["summary"]["rows"] == expected
        assert len({row["asset_id"] for row in plan["rows"]}) == expected


def test_pinned_axes_never_vary(payload):
    for plan in payload["plans"]:
        for axis, value in plan["pinned_axes"].items():
            assert {row["realized_attributes"][axis] for row in plan["rows"]} == {value}


def test_every_row_has_exactly_one_disposition(payload):
    for plan in payload["plans"]:
        for row in plan["rows"]:
            flags = (
                row["already_registered"],
                row["requires_new_ue_asset"],
                row["runtime_only_derivation"],
            )
            assert sum(bool(flag) for flag in flags) == 1, row


def test_runtime_only_rows_differ_from_their_source_only_by_scale(payload):
    scale_axes = {axis for axis, op in OPERATION_BY_AXIS.items() if op == TOOL.RUNTIME_ONLY_OPERATION}
    for plan in payload["plans"]:
        by_id = {row["asset_id"]: row for row in plan["rows"]}
        for row in plan["rows"]:
            if not row["runtime_only_derivation"]:
                assert row["derived_from"] is None
                continue
            parent = by_id[row["derived_from"]]
            differing = {
                axis
                for axis in plan["enabled_axes"]
                if row["realized_attributes"][axis] != parent["realized_attributes"][axis]
            }
            assert differing and differing <= scale_axes


def test_only_the_source_group_reuses_the_existing_ue_asset(payload):
    for plan in payload["plans"]:
        reused = [group for group in plan["ue_import_groups"] if group["reuses_existing_ue_asset"]]
        assert len(reused) == 1
        assert plan["summary"]["ue_imports_required"] == len(plan["ue_import_groups"]) - 1
        assert plan["summary"]["rows"] == (
            plan["summary"]["already_registered"]
            + plan["summary"]["ue_imports_required"]
            + plan["summary"]["runtime_only_derivations"]
        )


def test_coat_domain_comes_from_the_contract(payload):
    for plan in payload["plans"]:
        identity = plan["source_asset"]["identity"]
        key = (identity["species_id"], identity["breed_id"], plan["source_asset"]["coat_profile_id"])
        assert tuple(plan["axis_domains"]["coat_profile"]) == tuple(COAT_PROFILE_DOMAINS[key])
        assert tuple(plan["axis_domains"]["size"]) == tuple(CANONICAL_DOMAINS["size"])


def _first_eligible_entry():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for entry in registry["assets"]:
        realized = entry.get("realized_attributes") or {}
        if isinstance(realized.get("coat_profile"), dict):
            return copy.deepcopy(entry)
    raise AssertionError("the registry carries no appearance-realized asset")


def test_unregistered_breed_fails_closed():
    entry = _first_eligible_entry()
    entry["identity"] = dict(entry["identity"], breed_id="breed_that_is_not_reviewed")
    spec = TOOL._load_fanout_spec(SPEC_PATH)
    with pytest.raises(TOOL.PlanError, match="no reviewed coat domain"):
        TOOL.plan_for_entry(entry, spec)


def test_asset_id_without_encoded_attributes_fails_closed():
    entry = _first_eligible_entry()
    entry["asset_id"] = "asset_without_encoded_attributes_v1"
    spec = TOOL._load_fanout_spec(SPEC_PATH)
    with pytest.raises(TOOL.PlanError, match="does not encode its realized attributes"):
        TOOL.plan_for_entry(entry, spec)


def test_spec_with_an_undecided_axis_fails_closed(tmp_path):
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    spec["pinned_axes"] = {}
    path = tmp_path / "partial_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(TOOL.PlanError, match="leaves axes undecided"):
        TOOL._load_fanout_spec(path)
