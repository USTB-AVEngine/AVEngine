from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_TOOL_SPEC = importlib.util.spec_from_file_location(
    "merge_spear_apartment_render_shards",
    Path(__file__).resolve().parents[2]
    / "tools/dataset/merge_spear_apartment_render_shards.py",
)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(_TOOL)


def _scenario(scenario_id: str) -> dict:
    return {"scenario_id": scenario_id, "plan": {"episode": scenario_id}}


def _shard(
    root: Path,
    *,
    declared: list[dict],
    completed_ids: tuple[str, ...],
    execution_partition: dict | None = None,
) -> None:
    root.mkdir()
    plan = {"scenarios": declared}
    if execution_partition is not None:
        plan["execution_partition"] = execution_partition
    (root / "suite_execution_plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    for scenario_id in completed_ids:
        scenario_root = root / scenario_id
        scenario_root.mkdir()
        (scenario_root / "evidence.json").write_text(
            json.dumps({"status": "pass", "scenario_id": scenario_id}),
            encoding="utf-8",
        )


def test_collects_complete_union_and_reports_duplicate_scenarios(
    tmp_path: Path,
) -> None:
    full_scenarios = [_scenario("a"), _scenario("b"), _scenario("c")]
    first = tmp_path / "first"
    second = tmp_path / "second"
    _shard(
        first,
        declared=full_scenarios[:2],
        completed_ids=("a", "b"),
    )
    _shard(
        second,
        declared=full_scenarios[1:],
        completed_ids=("b", "c"),
    )

    selected, report = _TOOL._collect_scenario_sources(
        expected_episode_ids=("a", "b", "c"),
        full_suite_plan={"scenarios": full_scenarios},
        shard_roots=(first, second),
    )

    assert selected == {
        "a": first / "a",
        "b": first / "b",
        "c": second / "c",
    }
    assert report["duplicate_scenario_count"] == 1
    assert report["duplicate_occurrence_count"] == 1


def test_rejects_missing_or_changed_shard_scenario(tmp_path: Path) -> None:
    full_scenarios = [_scenario("a"), _scenario("b")]
    changed = tmp_path / "changed"
    _shard(
        changed,
        declared=[_scenario("a"), {"scenario_id": "b", "plan": {"changed": True}}],
        completed_ids=("a", "b"),
    )
    with pytest.raises(RuntimeError, match="differs from full plan"):
        _TOOL._collect_scenario_sources(
            expected_episode_ids=("a", "b"),
            full_suite_plan={"scenarios": full_scenarios},
            shard_roots=(changed,),
        )

    missing = tmp_path / "missing"
    _shard(
        missing,
        declared=full_scenarios,
        completed_ids=("a",),
    )
    with pytest.raises(RuntimeError, match="missing 1 scenarios"):
        _TOOL._collect_scenario_sources(
            expected_episode_ids=("a", "b"),
            full_suite_plan={"scenarios": full_scenarios},
            shard_roots=(missing,),
        )


def _partition(
    *,
    shard_count: int,
    shard_index: int,
    episode_ids: tuple[str, ...],
) -> dict:
    return {
        "kind": "contiguous_manifest_episode_ids",
        "shard_count": shard_count,
        "shard_index": shard_index,
        "total_episode_count": 4,
        "selected_episode_count": len(episode_ids),
        "first_episode_id": episode_ids[0],
        "last_episode_id": episode_ids[-1],
    }


def test_exact_partitions_require_complete_unique_shard_indices(
    tmp_path: Path,
) -> None:
    full_scenarios = [_scenario(value) for value in ("a", "b", "c", "d")]
    first = tmp_path / "first"
    second = tmp_path / "second"
    _shard(
        first,
        declared=full_scenarios[:2],
        completed_ids=("a", "b"),
        execution_partition=_partition(
            shard_count=2, shard_index=0, episode_ids=("a", "b")
        ),
    )
    _shard(
        second,
        declared=full_scenarios[2:],
        completed_ids=("c", "d"),
        execution_partition=_partition(
            shard_count=2, shard_index=1, episode_ids=("c", "d")
        ),
    )

    _selected, report = _TOOL._collect_scenario_sources(
        expected_episode_ids=("a", "b", "c", "d"),
        full_suite_plan={"scenarios": full_scenarios},
        shard_roots=(first, second),
    )
    assert [value["shard_index"] for value in report["execution_partitions"]] == [
        0,
        1,
    ]
    assert report["duplicate_scenario_count"] == 0

    duplicate_index = tmp_path / "duplicate_index"
    _shard(
        duplicate_index,
        declared=full_scenarios[2:],
        completed_ids=("c", "d"),
        execution_partition=_partition(
            shard_count=2, shard_index=0, episode_ids=("c", "d")
        ),
    )
    with pytest.raises(RuntimeError, match="differs from its exact partition"):
        _TOOL._collect_scenario_sources(
            expected_episode_ids=("a", "b", "c", "d"),
            full_suite_plan={"scenarios": full_scenarios},
            shard_roots=(first, duplicate_index),
        )
