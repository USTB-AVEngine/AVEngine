#!/usr/bin/env python3
"""Author the five audio_playback static_object candidate profiles.

Every constant that already exists upstream is read from the reviewed
microwave profile instead of being retyped, so the pinned model revisions
and the reviewed pose guard cannot drift away from the family they join.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SPEAR = Path("/data/jzy/code/SPEAR-lead-b")
PROFILE_DIR = (
    SPEAR / "data/controlled_source_attributes_v1/candidate_profiles/static_object"
)
REFERENCE = PROFILE_DIR / "kitchen_appliance_microwave_oven_product_view_v1.json"
REVISION = "2026_08_26_v1_audio_playback_base"

sys.path.insert(0, str(SPEAR))
from tools import controlled_source_asset_schema as contracts  # noqa: E402

reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
MODEL_REVISIONS = reference["generation_contract"]["model_revisions"]
POSE_GUARD = reference["generation_contract"]["pose_guard_prompt"]
BASE_NEGATIVE = reference["generation_contract"]["negative_prompt"]

CATEGORY_LABEL = "audio playback device"

# Failure modes that are specific to loudspeaker product photography: these
# objects are almost always shot as stereo pairs, usually with the grille on,
# and usually next to the gear that drives them.
SPEAKER_NEGATIVE = (
    "pair of speakers, stereo pair, two speakers, stacked speakers, "
    "speaker stand, tripod stand, wall bracket, subwoofer box, "
    "amplifier, receiver, turntable, cables, grille cloth over the drivers, "
    "removed front panel, exploded view, cutaway"
)


def profile(
    *,
    object_type: str,
    object_type_label: str,
    subject_label: str,
    form_factor: str,
    form_factor_label: str,
    material: str,
    material_label: str,
    sampled_attribute: str,
    sampled_label: str,
    sampled_values: dict[str, str],
    positive_template: str,
    extra_negative: str,
    height_cm: float,
    tolerance_cm: float,
    height_source: str,
    acoustic_profile_id: str,
    default_event_class: str,
    identification_question: str,
) -> dict:
    schema_id = f"audio_playback_{object_type}_product_view_v1"
    return {
        "schema": "avengine_attribute_profile_v1",
        "profile_schema_id": schema_id,
        "profile_revision": REVISION,
        "asset_class": "static_object",
        "lineage_group_id": "static_product_view_t2i_flux2_v1",
        "state_classification": "research_candidate",
        "taxonomy": {"category": "audio_playback", "object_type": object_type},
        "base_template": {
            "template_id": "static_object_text_prompt_only_v1",
            "kind": "text_prompt_only",
            "artifact": None,
            "provenance_status": "verified",
            "usage_scope": "research_candidate",
        },
        "fixed_attributes": {"form_factor": form_factor, "material": material},
        "sampled_attribute_domains": {sampled_attribute: list(sampled_values)},
        "forbidden_combinations": [],
        "generation_contract": {
            "route": "flux2_pixal3d_static_v1",
            "prompt_template_id": f"static_product_view_t2i_v1_{object_type}",
            "base_acquisition_policy": {
                "policy_id": "static_object_per_request_one_shot_v1",
                "acquisition_unit": "one_frozen_asset_per_request",
                "sampled_domains_must_be_singleton": False,
                "downstream_instance_route": "flux2_pixal3d_static_v1",
                "profile_validation": (
                    "all_predeclared_requests_count_zero_hidden_failures"
                ),
            },
            "positive_template": positive_template,
            "pose_guard_prompt": POSE_GUARD,
            "negative_prompt": f"{BASE_NEGATIVE}, {extra_negative}",
            "value_labels": {
                "category": {"audio_playback": CATEGORY_LABEL},
                "object_type": {object_type: object_type_label},
                "form_factor": {form_factor: form_factor_label},
                "material": {material: material_label},
                sampled_attribute: dict(sampled_values),
            },
            "model_revisions": dict(MODEL_REVISIONS),
        },
        "target_physical_profiles": {
            "profile_id": f"{schema_id}_physical_candidate_v1",
            "control_attribute": None,
            "measurement": "height_cm",
            "mode": "absolute_measurement",
            "reference_value_cm": height_cm,
            "reference_provenance": {
                "status": "provisional",
                "source_id": height_source,
                "artifact": None,
                "notes": (
                    "Provisional typical retail dimension; measure the approved "
                    "mesh before formal registration. Uniform height scaling "
                    "amplifies any aspect error from reconstruction, so check "
                    "the width and depth of the finalized mesh as well."
                ),
            },
            "values": {
                "fixed": {
                    "target_value_cm": height_cm,
                    "tolerance_cm": tolerance_cm,
                }
            },
        },
        "rig_profile": None,
        "acoustic_profile": {
            "profile_id": acoustic_profile_id,
            "default_event_class": default_event_class,
            "allowed_event_classes": [
                "music_playback",
                "speech_playback",
                "any_audioset_class_playback",
                "silent",
            ],
            "selection_attributes": ["category", "object_type"],
        },
        "locked_attributes": [
            "category",
            "object_type",
            "form_factor",
            "material",
        ],
        "qa_contract": {
            "subject_label": subject_label,
            "attributes": {
                sampled_attribute: {
                    "kind": "categorical",
                    "label": sampled_label,
                    "value_labels": dict(sampled_values),
                    "identification_question": identification_question,
                }
            },
        },
    }


WOOD_FINISHES = {
    "black_ash": "matte black ash",
    "walnut_veneer": "warm brown walnut",
    "white_satin": "satin white",
}

PROFILES = [
    profile(
        object_type="bookshelf_speaker",
        object_type_label="bookshelf loudspeaker",
        subject_label="bookshelf loudspeaker",
        form_factor="sealed_two_way_cabinet",
        form_factor_label="two-way cabinet",
        material="wood",
        material_label="wood-veneer",
        sampled_attribute="finish",
        sampled_label="cabinet finish",
        sampled_values=WOOD_FINISHES,
        positive_template=(
            "One {finish} {material} {form_factor} {object_type}, a household "
            "{category}: a compact upright rectangular loudspeaker box, clearly "
            "taller than it is wide, standing on its own flat plinth. The front "
            "baffle carries exactly two exposed round drivers on the vertical "
            "centerline: one large woofer cone in the lower half and one small "
            "dome tweeter above it, with a single circular port opening below "
            "the woofer. The side, top and back panels are plain and flat with "
            "no controls, no cloth and no visible screws."
        ),
        extra_negative=(
            f"{SPEAKER_NEGATIVE}, bookshelf, books, shelf, single driver, "
            "three or more drivers, horizontal orientation, lying down"
        ),
        height_cm=33,
        tolerance_cm=6,
        height_source="bookshelf_loudspeaker_typical_retail_dimension_v1",
        acoustic_profile_id="loudspeaker_playback_v1",
        default_event_class="music_playback",
        identification_question=(
            "What is the cabinet finish of {instance_label}?"
        ),
    ),
    profile(
        object_type="floorstanding_speaker",
        object_type_label="floorstanding loudspeaker",
        subject_label="floorstanding loudspeaker",
        form_factor="three_way_floor_tower",
        form_factor_label="three-way tower",
        material="wood",
        material_label="wood-veneer",
        sampled_attribute="finish",
        sampled_label="cabinet finish",
        sampled_values=WOOD_FINISHES,
        positive_template=(
            "One {finish} {material} {form_factor} {object_type}, a household "
            "{category}: a tall slim upright loudspeaker column resting directly "
            "on the floor on a rectangular plinth, roughly five times taller "
            "than it is wide. The front baffle carries four exposed round "
            "drivers stacked on the vertical centerline: two large woofer cones "
            "low down, one midrange cone above them, and one small dome tweeter "
            "at the very top. The side and back panels are plain and flat with "
            "no controls and no cloth."
        ),
        extra_negative=(
            f"{SPEAKER_NEGATIVE}, short cabinet, tabletop speaker, "
            "single driver, two drivers, horizontal orientation, lying down"
        ),
        height_cm=100,
        tolerance_cm=15,
        height_source="floorstanding_loudspeaker_typical_retail_dimension_v1",
        acoustic_profile_id="loudspeaker_playback_v1",
        default_event_class="music_playback",
        identification_question=(
            "What is the cabinet finish of {instance_label}?"
        ),
    ),
    profile(
        object_type="soundbar",
        object_type_label="soundbar",
        subject_label="soundbar",
        form_factor="wide_low_bar",
        form_factor_label="wide low bar",
        material="plastic",
        material_label="moulded plastic",
        sampled_attribute="finish",
        sampled_label="body finish",
        sampled_values={
            "matte_black": "matte black",
            "light_gray_fabric": "light gray fabric",
            "white_satin": "satin white",
        },
        positive_template=(
            "One {finish} {material} {form_factor} {object_type}, a household "
            "{category}: one single long horizontal bar at least ten times wider "
            "than it is tall, lying flat on a surface on two small low feet. Its "
            "entire front face is one continuous perforated speaker grille "
            "running the full width, and its top is smooth with one small row of "
            "flush touch controls at the right end. The ends are rounded and "
            "plain."
        ),
        extra_negative=(
            f"{SPEAKER_NEGATIVE}, television, screen, monitor, display panel, "
            "separate subwoofer, tall cabinet, square proportions, "
            "vertical orientation, mounted under a television"
        ),
        height_cm=7,
        tolerance_cm=2,
        height_source="soundbar_typical_retail_dimension_v1",
        acoustic_profile_id="loudspeaker_playback_v1",
        default_event_class="speech_playback",
        identification_question="What is the body finish of {instance_label}?",
    ),
    profile(
        object_type="smart_speaker",
        object_type_label="smart speaker",
        subject_label="smart speaker",
        form_factor="fabric_wrapped_cylinder",
        form_factor_label="fabric-wrapped cylinder",
        material="textile",
        material_label="woven textile",
        sampled_attribute="finish",
        sampled_label="wrap color",
        sampled_values={
            "charcoal": "charcoal gray",
            "light_gray": "light gray",
            "sandstone": "sandstone beige",
        },
        positive_template=(
            "One {finish} {material} {form_factor} {object_type}, a household "
            "{category}: a short upright cylinder roughly as tall as it is wide, "
            "wrapped from its base to its shoulder in one seamless acoustically "
            "open mesh weave with no seam and no pattern. Its flat top is a "
            "plain hard disc carrying four small round openings near the rim and "
            "nothing else. A slim dark ring forms the base. Nothing protrudes "
            "from the body."
        ),
        extra_negative=(
            f"{SPEAKER_NEGATIVE}, screen, display, touchscreen, camera lens, "
            "sphere, ball, cone, hourglass shape, handle, buttons on the side, "
            "charging dock"
        ),
        height_cm=17,
        tolerance_cm=5,
        height_source="smart_speaker_typical_retail_dimension_v1",
        acoustic_profile_id="loudspeaker_playback_v1",
        default_event_class="speech_playback",
        identification_question="What is the wrap color of {instance_label}?",
    ),
    profile(
        object_type="television",
        object_type_label="television",
        subject_label="television",
        form_factor="flat_panel_16_9",
        form_factor_label="16:9 flat panel",
        material="plastic",
        material_label="moulded plastic",
        sampled_attribute="stand_type",
        sampled_label="stand type",
        sampled_values={
            "central_pedestal": "single central pedestal",
            "two_splayed_feet": "two splayed feet at the outer edges",
        },
        positive_template=(
            "One {material} {form_factor} {object_type} standing on a "
            "{stand_type}, a household {category}: one large thin flat-screen "
            "display much wider than it is tall, with a very narrow dark bezel "
            "on all four sides and a completely blank matte dark screen showing "
            "nothing at all. The back is plain and flat. It stands on a table, "
            "not on a wall."
        ),
        extra_negative=(
            "wall mounted, wall bracket, soundbar, remote control, "
            "screen content, picture on screen, image on screen, bright screen, "
            "glowing screen, curved screen, computer monitor, keyboard, "
            "reflection on screen, media console, shelf unit, "
            f"{SPEAKER_NEGATIVE}"
        ),
        height_cm=72,
        tolerance_cm=10,
        height_source="fifty_five_inch_television_typical_retail_dimension_v1",
        acoustic_profile_id="television_playback_v1",
        default_event_class="speech_playback",
        identification_question="What stand does {instance_label} stand on?",
    ),
]


def main() -> int:
    written = []
    for payload in PROFILES:
        contracts.validate_attribute_profile(payload)
        combinations = contracts.legal_attribute_combinations(payload)
        path = PROFILE_DIR / f"{payload['profile_schema_id']}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        written.append((path.name, len(combinations)))
    for name, count in written:
        print(f"ok  {name}  legal_combinations={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
