from pathlib import Path

from avengine.contracts.json_io import load_json
from avengine.m6x.asset_emitter import validate_asset_emitter_binding_set
from avengine.runtime_profiles import load_default_source_asset_runtime_registry
from tools.m6x.select_asset_bound_trajectories import _pair_templates


ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "examples/m7/apartment_generated_asset_pair_templates.json"
BORDER_COLLIE = (
    "generated_border_collie_black_white_medium_standard_adult_research_v1"
)
HUMAN = "rocketbox_human_male_adult_01_m5_1_candidate"
CAT = "generated_abyssinian_ruddy_medium_standard_adult_research_v1"


def test_generated_asset_pair_templates_prove_ordered_slot_replacement():
    value = load_json(TEMPLATES)
    templates = _pair_templates(
        value,
        source_registry=load_default_source_asset_runtime_registry(),
    )
    pairs = set()
    border_collie_offsets = set()
    cat_offsets = set()
    slots_by_asset = {HUMAN: set(), BORDER_COLLIE: set(), CAT: set()}
    for _pair_id, binding_set in templates:
        bindings = validate_asset_emitter_binding_set(binding_set)
        pair = tuple(bindings[slot].asset_id for slot in ("source1", "source2"))
        pairs.add(pair)
        for slot, binding in bindings.items():
            slots_by_asset[binding.asset_id].add(slot)
            if binding.asset_id == BORDER_COLLIE:
                border_collie_offsets.add(binding.emitter_offset_m)
                assert binding.local_anatomical_forward_axis == (1.0, 0.0, 0.0)
            if binding.asset_id == CAT:
                cat_offsets.add(binding.emitter_offset_m)
                assert binding.local_anatomical_forward_axis == (1.0, 0.0, 0.0)
    assert pairs == {
        (HUMAN, BORDER_COLLIE),
        (BORDER_COLLIE, HUMAN),
        (CAT, BORDER_COLLIE),
        (BORDER_COLLIE, CAT),
    }
    assert slots_by_asset == {
        HUMAN: {"source1", "source2"},
        BORDER_COLLIE: {"source1", "source2"},
        CAT: {"source1", "source2"},
    }
    assert border_collie_offsets == {
        (0.40465284499021825, 0.6465226403698251, 0.0)
    }
    assert cat_offsets == {
        (0.38869346364905827, 0.16641961991328985, 0.0)
    }
