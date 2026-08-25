from pathlib import Path

from avengine.contracts.json_io import load_json
from avengine.routes.asset_emitter import validate_asset_emitter_binding_set
from avengine.runtime_profiles import load_default_source_asset_runtime_registry
from tools.routes.select_asset_bound_trajectories import _pair_templates


ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "examples/m7/apartment_generated_asset_pair_templates.json"
CURRENT_DOG = (
    "generated_pembroke_welsh_corgi_red_white_medium_standard_adult_research_v1"
)
HUMAN = "rocketbox_human_male_adult_01_m5_1_candidate"
CURRENT_CAT = (
    "generated_british_shorthair_blue_medium_stocky_adult_research_v1"
)


def test_generated_asset_pair_templates_prove_ordered_slot_replacement():
    value = load_json(TEMPLATES)
    templates = _pair_templates(
        value,
        source_registry=load_default_source_asset_runtime_registry(),
    )
    pairs = set()
    current_dog_offsets = set()
    cat_offsets = set()
    slots_by_asset = {HUMAN: set(), CURRENT_DOG: set(), CURRENT_CAT: set()}
    for _pair_id, binding_set in templates:
        bindings = validate_asset_emitter_binding_set(binding_set)
        pair = tuple(bindings[slot].asset_id for slot in ("source1", "source2"))
        pairs.add(pair)
        for slot, binding in bindings.items():
            slots_by_asset[binding.asset_id].add(slot)
            if binding.asset_id == CURRENT_DOG:
                current_dog_offsets.add(binding.emitter_offset_m)
                assert binding.local_anatomical_forward_axis == (1.0, 0.0, 0.0)
            if binding.asset_id == CURRENT_CAT:
                cat_offsets.add(binding.emitter_offset_m)
                assert binding.local_anatomical_forward_axis == (1.0, 0.0, 0.0)
    assert pairs == {
        (HUMAN, CURRENT_DOG),
        (CURRENT_DOG, HUMAN),
        (CURRENT_CAT, CURRENT_DOG),
        (CURRENT_DOG, CURRENT_CAT),
    }
    assert slots_by_asset == {
        HUMAN: {"source1", "source2"},
        CURRENT_DOG: {"source1", "source2"},
        CURRENT_CAT: {"source1", "source2"},
    }
    assert current_dog_offsets == {
        (0.2540588796012138, 0.30917552322485176, 0.0)
    }
    assert cat_offsets == {
        (0.2503672200051257, 0.23847342533548063, 0.0)
    }
