"""The prompt budget checker must enumerate every value and fail closed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY_ROOT / "tools/assets/check_prompt_token_budget.py"
SPEC_PATH = REPOSITORY_ROOT / "examples/assets/prompt_budget_v1.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("check_prompt_token_budget", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


class WordTokenizer:
    """Stand-in for the model tokenizer: one token per whitespace-separated word."""

    def __call__(self, text: str) -> dict[str, list[str]]:
        return {"input_ids": text.split()}


def _profile(**overrides):
    profile = {
        "profile_schema_id": "family_breed_rest_side_v1",
        "profile_revision": "test_v1",
        "taxonomy": {"species": "species_token", "breed": "breed_token"},
        "fixed_attributes": {"coat_length": "short"},
        "sampled_attribute_domains": {"size": ["small", "medium", "large"], "coat_color": ["a", "b"]},
        "generation_contract": {
            "positive_template": "one {size} {coat_length} {coat_color} {breed} {species}",
            "pose_guard_prompt": "guard " * 5,
            "negative_prompt": "no " * 4,
            "value_labels": {"breed": {"breed_token": "Labelled Breed"}},
        },
    }
    profile.update(overrides)
    return profile


def test_every_sampled_combination_is_checked():
    measured = TOOL.measure_profile(_profile(), WordTokenizer(), 512)
    assert measured["combinations_checked"] == 6
    seen = {
        (item["sampled_values"]["size"], item["sampled_values"]["coat_color"])
        for item in measured["combinations"]
    }
    assert len(seen) == 6


def test_value_labels_are_applied():
    measured = TOOL.measure_profile(_profile(), WordTokenizer(), 512)
    assert measured["fits"]
    # "one <size> <coat_length> <coat_color> Labelled Breed <species>" is seven words:
    # the two-word label lengthens the rendered positive by one token.
    assert measured["worst_case"]["positive_tokens"] == 7


def test_overflow_is_reported_per_combination():
    measured = TOOL.measure_profile(_profile(), WordTokenizer(), 8)
    assert not measured["fits"]
    worst = measured["worst_case"]
    assert worst["overflow_tokens"] == worst["effective_tokens"] - 8
    assert worst["overflow_tokens"] > 0


def test_worst_case_is_the_longest_combination():
    profile = _profile()
    profile["sampled_attribute_domains"]["coat_color"] = ["a", "a much longer coat name"]
    measured = TOOL.measure_profile(profile, WordTokenizer(), 512)
    assert measured["worst_case"]["sampled_values"]["coat_color"] == "a much longer coat name"


@pytest.mark.parametrize("field", ["positive_template", "pose_guard_prompt", "negative_prompt"])
def test_missing_prompt_field_fails_closed(field):
    profile = _profile()
    profile["generation_contract"][field] = "   "
    with pytest.raises(TOOL.BudgetError, match=field):
        TOOL.measure_profile(profile, WordTokenizer(), 512)


def test_empty_sampled_domain_fails_closed():
    profile = _profile()
    profile["sampled_attribute_domains"]["size"] = []
    with pytest.raises(TOOL.BudgetError, match="is empty"):
        TOOL.measure_profile(profile, WordTokenizer(), 512)


def test_placeholder_without_a_value_fails_closed():
    profile = _profile()
    profile["generation_contract"]["positive_template"] += " {unknown_axis}"
    with pytest.raises(TOOL.BudgetError, match="without a value"):
        TOOL.measure_profile(profile, WordTokenizer(), 512)


def test_shipped_budget_is_arithmetically_possible():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    caps = spec["caps_tokens"]
    parts = caps["positive_template_rendered"] + caps["pose_guard_prompt"] + caps["negative_prompt"]
    assert parts < caps["effective_prompt"], "the per-part caps cannot exceed the window"
    assert caps["effective_prompt"] == spec["model_window"]["max_sequence_length"]
    assert TOOL.DEFAULT_MAX_SEQUENCE_LENGTH == caps["effective_prompt"]


def test_shipped_shared_blocks_are_present_and_referenced():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    blocks = spec["shared_blocks"]
    assert blocks["pose_guard_prompt"].strip()
    assert blocks["negative_prompt"].strip()
    assert "check_prompt_token_budget.py" in spec["authoring_rule"]


def test_effective_prompt_format_matches_the_worker_concatenation():
    rendered = TOOL.EFFECTIVE_PROMPT_FORMAT.format(prompt="P", negative="N")
    assert rendered == "P Avoid: N."
