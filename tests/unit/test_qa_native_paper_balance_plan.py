from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPOSITORY
    / "schemas"
    / "avengine_native_paper_balance_episode_plan_v1.schema.json"
)
PLAN_PATH = REPOSITORY / "examples/qa/native_paper_balance_episode_plan_v1.json"
GPU_REVISION_SCHEMA_PATH = (
    REPOSITORY
    / "schemas"
    / "avengine_native_paper_balance_gpu_plan_revision_v1.schema.json"
)
GPU_REVISION_PATH = (
    REPOSITORY / "examples/qa/native_paper_balance_gpu1_revision_v1.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_native_paper_balance_plan_is_exact_and_valid() -> None:
    plan = _load(PLAN_PATH)
    jsonschema.Draft202012Validator(_load(SCHEMA_PATH)).validate(plan)
    assert [episode["variant"] for episode in plan["episodes"]] == [
        "stationary_source2_first",
        "source2_right_entry_second_transcript",
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_authority", {"renderer": "synthetic"}),
        (
            "gpu_policy",
            {
                "physical_gpu_index": 0,
                "require_idle_before_launch": False,
                "forbidden_gpu_indices": [0, 3],
            },
        ),
    ],
)
def test_native_paper_balance_plan_rejects_non_native_or_unsafe_substitution(
    field: str, replacement: object
) -> None:
    plan = deepcopy(_load(PLAN_PATH))
    plan[field] = replacement
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_load(SCHEMA_PATH)).validate(plan)


def test_gpu1_revision_preserves_gpu2_plan_and_forbids_gpu0_and_gpu3() -> None:
    revision = _load(GPU_REVISION_PATH)
    jsonschema.Draft202012Validator(_load(GPU_REVISION_SCHEMA_PATH)).validate(
        revision
    )
    assert revision["superseded_gpu_policy"]["physical_gpu_index"] == 2
    assert revision["active_gpu_policy"] == {
        "physical_gpu_index": 1,
        "require_idle_before_launch": True,
        "forbidden_gpu_indices": [0, 3],
    }
