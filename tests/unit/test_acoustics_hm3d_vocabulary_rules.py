"""The v3 residential ruleset must resolve HM3D's vocabulary, traps included.

The ruleset's hints are substring matches where order is precedence, so the
dangerous cases are compounds: "toilet_paper" must be paper before the ceramic
"toilet" hint can see it, "bookcase" wood before the fabric "case", and a bare
"unknown" object must stay on the default candidates, because an unknown
object's honest prior is the default mix - a greedy hint for it once captured
MP3D's unknown_object and this suite caught it. These are compiled through the
public API so what is asserted is what packages actually receive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.acoustics.semantic_materials import (
    SemanticSurfaceIdentity,
    compile_semantic_material_documents,
)

RULES_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/acoustics/semantic_materials/residential_material_rules.json"
)


def _decisions(categories: list[str]) -> dict[str, dict]:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    compiled = compile_semantic_material_documents(
        room_id="unit_hm3d_vocabulary",
        surfaces=[
            SemanticSurfaceIdentity(
                source_material_name=category,
                semantic_category=category,
                identity_key=f"unit_hm3d_vocabulary/{category}",
                object_name=category,
            )
            for category in categories
        ],
        rules=rules,
        seed=20260827,
        source_to_canonical={
            "matrix_row_major": [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            "source": "unit test identity",
            "reviewed": True,
        },
    )
    return {
        decision["semantic_category"]: decision
        for decision in compiled.report["decisions"]
    }


# label -> the material family its matched hint must offer
_TRAPS = {
    "toilet_paper": {"paper_stack"},
    "bath_mat": {"carpet", "carpet_heavy"},
    "toilet_seat": {"ceramic", "ceramic_tile"},
    "bookcase": {"wood_thin", "wood_thick"},
    "staircase": {"wood_thick", "concrete"},
    "iron_board": {"wood_thin"},
    "railing": {"wood_thick", "steel"},
    "laundry_basket": {"wood_thin", "fabric"},
    "led_tv": {"glass", "generic_hard_surface"},
    "nightstand": {"wood_thin", "wood_thick"},
    "unannotated": {"gypsum_board", "plaster", "wood_thick"},
    "balustrade": {"wood_thick", "steel"},
    "computer_desk": {"wood_thin", "wood_thick"},
}


def test_hm3d_compound_labels_hit_the_intended_material_family() -> None:
    decisions = _decisions(sorted(_TRAPS))
    for label, family in _TRAPS.items():
        decision = decisions[label]
        offered = {
            candidate["material"] for candidate in decision["candidate_materials"]
        }
        assert decision["resolution"] == "name_hint", (label, decision["resolution"])
        assert offered == family, (label, offered)


def test_an_unknown_object_keeps_the_default_prior() -> None:
    """A hint greedy enough to catch bare "unknown" is a judgement error."""

    decisions = _decisions(["unknown", "unknown_object"])
    for label in ("unknown", "unknown_object"):
        assert decisions[label]["resolution"] == "default_candidate", label


def test_the_hm3d_val_00800_vocabulary_mostly_resolves() -> None:
    """The 74 categories that fell to default at 00800 must now be rare."""

    fell_through_at_00800 = [
        "alarm", "alarm_control", "backpack", "bag", "balustrade", "basket",
        "bath", "bath_mat", "bath_sink", "bathroom_accessory", "bicycle",
        "blanket", "board", "bottle_of_soap", "box", "boxes", "briefcase",
        "bucket", "case", "coffee_machine", "computer_desk", "electric_box",
        "folder", "fruit_bowl", "handbag", "iron", "iron_board",
        "knife_holder", "laundry_basket", "led_tv", "nightstand", "pad",
        "pc_tower", "pillar", "plush_toy", "printer", "shoe", "speaker",
        "stack_of_papers", "storage_box", "tap", "tissue_box", "toilet_brush",
        "toilet_paper", "toilet_seat", "toy", "trashcan", "tv", "unknown",
        "wall_clock", "worktop",
    ]
    decisions = _decisions(fell_through_at_00800)
    defaults = sorted(
        label
        for label, decision in decisions.items()
        if decision["resolution"] == "default_candidate"
    )
    # "unknown" and bare "board" stay on the default prior by design.
    assert set(defaults) <= {"unknown", "board"}, defaults
