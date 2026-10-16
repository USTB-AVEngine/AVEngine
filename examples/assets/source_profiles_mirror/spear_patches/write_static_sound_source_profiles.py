#!/usr/bin/env python3
"""Author the first-pass form profiles for the 14 static sound-source families.

The four profiles that already existed before this work order are copied byte
for byte into the AVEngine mirror and are never regenerated here.  The other
24 profiles cover every remaining form factor in the owner-approved list, one
default finish per form.  Shared model revisions and product-view prompts come
from an audio_playback profile that has already passed the real FLUX.2 route.

The SPEAR checkout is an execution workspace with broken Git metadata.  This
script writes the execution copy and the versioned AVEngine mirror together and
refuses to replace either side when an existing file differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REVISION = "2026_08_26_v1_static_sound_forms"
EXISTING_PROFILE_NAMES = (
    "kitchen_appliance_microwave_oven_product_view_v1.json",
    "household_clock_alarm_clock_product_view_v1.json",
    "door_hardware_doorbell_chime_unit_product_view_v1.json",
    "communication_device_desk_telephone_product_view_v1.json",
)


def spec(
    *,
    category: str,
    category_label: str,
    object_type: str,
    object_label: str,
    form_factor: str,
    form_label: str,
    material: str,
    material_label: str,
    body_color: str,
    color_label: str,
    description: str,
    extra_negative: str,
    height_cm: float,
    tolerance_cm: float,
    acoustic_profile_id: str,
    default_event_class: str,
    allowed_event_classes: tuple[str, ...],
) -> dict[str, Any]:
    return locals()


SPECS = (
    spec(
        category="climate_control",
        category_label="climate-control appliance",
        object_type="air_conditioner",
        object_label="air conditioner",
        form_factor="wall_split",
        form_label="wall-mounted split indoor unit",
        material="plastic",
        material_label="moulded plastic",
        body_color="white",
        color_label="matte white",
        description=(
            "a wide shallow rectangular indoor unit with a flat back and one "
            "continuous adjustable discharge louver across the lower front; "
            "the dark air-outlet slot behind that louver is clearly visible"
        ),
        extra_negative="outdoor condenser, exposed fan, window unit, floor unit, wall, room scene",
        height_cm=30,
        tolerance_cm=6,
        acoustic_profile_id="air_conditioner_v1",
        default_event_class="air_conditioning",
        allowed_event_classes=("air_conditioning", "silent"),
    ),
    spec(
        category="climate_control",
        category_label="climate-control appliance",
        object_type="air_conditioner",
        object_label="air conditioner",
        form_factor="window_unit",
        form_label="window air-conditioning unit",
        material="steel",
        material_label="painted sheet steel",
        body_color="white",
        color_label="matte white",
        description=(
            "a compact deep box with a large front intake grille, a separate "
            "row of directional outlet louvers across the upper front, and a "
            "small control cluster confined to one side"
        ),
        extra_negative="split unit, outdoor condenser, portable floor unit, window frame, wall, room scene",
        height_cm=38,
        tolerance_cm=7,
        acoustic_profile_id="air_conditioner_v1",
        default_event_class="air_conditioning",
        allowed_event_classes=("air_conditioning", "silent"),
    ),
    spec(
        category="climate_control",
        category_label="climate-control appliance",
        object_type="air_conditioner",
        object_label="air conditioner",
        form_factor="portable_floor",
        form_label="portable floor air conditioner",
        material="plastic",
        material_label="moulded plastic",
        body_color="white",
        color_label="matte white",
        description=(
            "a tall narrow wheeled floor cabinet with one broad horizontal air "
            "outlet and adjustable louver on the upper front, a plain lower "
            "body, and four small integrated caster wheels"
        ),
        extra_negative="wall-mounted unit, window unit, outdoor condenser, loose exhaust hose, room scene",
        height_cm=70,
        tolerance_cm=10,
        acoustic_profile_id="air_conditioner_v1",
        default_event_class="air_conditioning",
        allowed_event_classes=("air_conditioning", "silent"),
    ),
    spec(
        category="kitchen_appliance",
        category_label="kitchen appliance",
        object_type="microwave_oven",
        object_label="microwave oven",
        form_factor="over_range",
        form_label="over-range microwave",
        material="steel",
        material_label="stainless sheet-steel body",
        body_color="silver",
        color_label="brushed stainless silver",
        description=(
            "a wide built-in microwave with a large dark front door, a narrow "
            "control panel on the right, and one clearly visible perforated "
            "ventilation grille running along the lower front edge"
        ),
        extra_negative="countertop microwave, cabinets, cooktop, kitchen scene, open door, food",
        height_cm=43,
        tolerance_cm=7,
        acoustic_profile_id="microwave_oven_v1",
        default_event_class="microwave_hum",
        allowed_event_classes=("microwave_hum", "microwave_beep", "silent"),
    ),
    spec(
        category="office_device",
        category_label="office device",
        object_type="printer",
        object_label="printer",
        form_factor="desktop_inkjet",
        form_label="desktop inkjet printer",
        material="plastic",
        material_label="moulded plastic",
        body_color="matte_black",
        color_label="matte black",
        description=(
            "a low compact desktop inkjet printer with a closed top lid, one "
            "integrated front paper-output slot and shallow tray, and a row of "
            "small motor ventilation slots beside the output opening"
        ),
        extra_negative="office copier, tall floor machine, loose paper, ink bottles, open service panels",
        height_cm=20,
        tolerance_cm=5,
        acoustic_profile_id="printer_v1",
        default_event_class="printer",
        allowed_event_classes=("printer", "silent"),
    ),
    spec(
        category="office_device",
        category_label="office device",
        object_type="printer",
        object_label="printer",
        form_factor="office_laser_mfp",
        form_label="office laser multifunction printer",
        material="plastic",
        material_label="moulded plastic",
        body_color="white",
        color_label="office white",
        description=(
            "a tall boxy multifunction laser printer with a flatbed scanner on "
            "top, a central recessed paper-output bay facing forward, two lower "
            "paper drawers, and visible ventilation slots beside the output bay"
        ),
        extra_negative="desktop inkjet, freestanding human-height copier, loose paper, open panels, office scene",
        height_cm=55,
        tolerance_cm=10,
        acoustic_profile_id="printer_v1",
        default_event_class="printer",
        allowed_event_classes=("printer", "silent"),
    ),
    spec(
        category="kitchen_appliance",
        category_label="kitchen appliance",
        object_type="blender",
        object_label="blender",
        form_factor="jug_blender",
        form_label="jug blender",
        material="plastic_and_glass",
        material_label="plastic motor base and clear glass jug",
        body_color="black",
        color_label="matte black",
        description=(
            "a clear lidded pitcher locked onto a broad motor base, with the "
            "blade hub visible inside the empty jug and one ring of narrow motor "
            "cooling vents clearly visible around the lower front of the base"
        ),
        extra_negative="food, liquid, fruit, detached jug, extra cup, immersion blender, hand mixer",
        height_cm=40,
        tolerance_cm=7,
        acoustic_profile_id="blender_v1",
        default_event_class="blender",
        allowed_event_classes=("blender", "silent"),
    ),
    spec(
        category="kitchen_appliance",
        category_label="kitchen appliance",
        object_type="blender",
        object_label="blender",
        form_factor="bullet_blender",
        form_label="compact bullet blender",
        material="plastic_and_steel",
        material_label="plastic cup and stainless motor base",
        body_color="silver",
        color_label="brushed silver",
        description=(
            "one narrow transparent blending cup mounted upside down and locked "
            "onto a short cylindrical motor base, with a continuous ring of "
            "small cooling slots visible near the bottom of the base"
        ),
        extra_negative="multiple cups, accessory lids, detached blade, jug blender, food, liquid, packaging",
        height_cm=35,
        tolerance_cm=7,
        acoustic_profile_id="blender_v1",
        default_event_class="blender",
        allowed_event_classes=("blender", "silent"),
    ),
    spec(
        category="household_clock",
        category_label="household clock",
        object_type="alarm_clock",
        object_label="alarm clock",
        form_factor="digital_cube",
        form_label="digital cube alarm clock",
        material="plastic",
        material_label="moulded plastic",
        body_color="black",
        color_label="matte black",
        description=(
            "a small rectangular bedside clock with a blank dark front display, "
            "two low feet, and a clearly visible perforated buzzer grille on the "
            "upper rear-facing bevel while the grille remains in view"
        ),
        extra_negative="analog clock face, twin bells, wall clock, digits, glowing text, phone, radio antenna",
        height_cm=8,
        tolerance_cm=3,
        acoustic_profile_id="alarm_clock_v1",
        default_event_class="alarm_clock",
        allowed_event_classes=("alarm_clock", "buzzer", "silent"),
    ),
    spec(
        category="door_hardware",
        category_label="door hardware",
        object_type="doorbell_chime_unit",
        object_label="doorbell",
        form_factor="video_doorbell",
        form_label="slim video doorbell",
        material="plastic",
        material_label="weather-resistant plastic",
        body_color="black",
        color_label="matte black",
        description=(
            "a slim vertical doorbell body with one camera lens at the top, one "
            "large circular illuminated push button near the bottom, and a "
            "separate cluster of small speaker-grille holes clearly visible on "
            "the front between them"
        ),
        extra_negative="door, wall, hand, detached mounting plate, chime box, keypad lock, intercom handset",
        height_cm=13,
        tolerance_cm=3,
        acoustic_profile_id="doorbell_v1",
        default_event_class="doorbell",
        allowed_event_classes=("doorbell", "ding_dong", "chime", "silent"),
    ),
    spec(
        category="communication_device",
        category_label="communication device",
        object_type="landline_phone",
        object_label="landline telephone",
        form_factor="wall_mounted",
        form_label="wall-mounted corded telephone",
        material="plastic",
        material_label="hard plastic",
        body_color="beige",
        color_label="warm beige",
        description=(
            "a narrow vertical telephone base with a corded handset seated in "
            "its side cradle, a front keypad, and a clearly visible horizontal "
            "ringer grille across the lower front of the base"
        ),
        extra_negative="desk phone, mobile phone, detached handset, floating handset, wall, room scene, loose cable",
        height_cm=23,
        tolerance_cm=5,
        acoustic_profile_id="landline_phone_v1",
        default_event_class="telephone_bell_ringing",
        allowed_event_classes=(
            "telephone",
            "telephone_bell_ringing",
            "telephone_dialing_dtmf",
            "dial_tone",
            "busy_signal",
            "silent",
        ),
    ),
    spec(
        category="communication_device",
        category_label="communication device",
        object_type="cellphone",
        object_label="cellphone",
        form_factor="bar_smartphone",
        form_label="bar smartphone",
        material="glass_and_metal",
        material_label="glass and metal",
        body_color="black",
        color_label="gloss black",
        description=(
            "a slim modern bar smartphone standing upright in portrait "
            "orientation on its flat bottom edge with a completely blank dark "
            "screen; one narrow earpiece speaker slit is clearly visible in the "
            "front bezel directly above the screen"
        ),
        extra_negative=(
            "screen content, text, app icons, hand, case, charger, cable, "
            "second phone, folding phone, lying flat, horizontal phone, stand"
        ),
        height_cm=15.5,
        tolerance_cm=2,
        acoustic_profile_id="cellphone_v1",
        default_event_class="ringtone",
        allowed_event_classes=("ringtone", "cellphone_vibration_alert", "silent"),
    ),
    spec(
        category="safety_device",
        category_label="household safety device",
        object_type="smoke_detector",
        object_label="smoke detector",
        form_factor="ceiling_disc",
        form_label="ceiling-mounted detector disc",
        material="plastic",
        material_label="moulded plastic",
        body_color="white",
        color_label="matte white",
        description=(
            "a low round detector disc resting on its flat mounting back, with "
            "a raised central test button and a complete ring of narrow sounder "
            "and smoke-entry slots clearly visible across the front face"
        ),
        extra_negative="ceiling, room scene, fire sprinkler, security camera, light fixture, separate mounting plate",
        height_cm=5,
        tolerance_cm=2,
        acoustic_profile_id="smoke_detector_v1",
        default_event_class="smoke_alarm",
        allowed_event_classes=("smoke_alarm", "fire_alarm", "silent"),
    ),
    spec(
        category="safety_device",
        category_label="household safety device",
        object_type="smoke_detector",
        object_label="smoke detector",
        form_factor="wall_square",
        form_label="square wall-mounted detector",
        material="plastic",
        material_label="moulded plastic",
        body_color="white",
        color_label="matte white",
        description=(
            "a compact square wall detector with rounded corners, a small test "
            "button, and a large field of unmistakable sounder slots on the "
            "front face, standing upright on its flat lower edge"
        ),
        extra_negative="wall, room scene, fire sprinkler, security camera, light fixture, blank featureless cover",
        height_cm=15,
        tolerance_cm=3,
        acoustic_profile_id="smoke_detector_v1",
        default_event_class="smoke_alarm",
        allowed_event_classes=("smoke_alarm", "fire_alarm", "silent"),
    ),
    spec(
        category="plumbing_fixture",
        category_label="plumbing fixture",
        object_type="toilet",
        object_label="toilet",
        form_factor="floor_close_coupled",
        form_label="floor close-coupled toilet",
        material="ceramic",
        material_label="glazed ceramic",
        body_color="white",
        color_label="gloss white",
        description=(
            "a floor-standing toilet with an integrated rear cistern, the seat "
            "and lid raised together, and the bowl viewed from above enough that "
            "the continuous flush-water slot beneath the inner rear rim is "
            "clearly visible"
        ),
        extra_negative="bathroom scene, wall-hung bowl, detached tank, bidet, closed lid, water splash, toilet paper",
        height_cm=78,
        tolerance_cm=10,
        acoustic_profile_id="toilet_v1",
        default_event_class="toilet_flush",
        allowed_event_classes=("toilet_flush", "silent"),
    ),
    spec(
        category="plumbing_fixture",
        category_label="plumbing fixture",
        object_type="toilet",
        object_label="toilet",
        form_factor="wall_hung",
        form_label="wall-hung toilet bowl",
        material="ceramic",
        material_label="glazed ceramic",
        body_color="white",
        color_label="gloss white",
        description=(
            "a compact wall-hung toilet bowl with a flat rear mounting face and "
            "no visible cistern, the seat and lid raised, and the flush-water "
            "channel beneath the inner rear rim clearly visible from above"
        ),
        extra_negative="bathroom scene, floor pedestal, exposed cistern, detached tank, bidet, closed lid, wall",
        height_cm=40,
        tolerance_cm=7,
        acoustic_profile_id="toilet_v1",
        default_event_class="toilet_flush",
        allowed_event_classes=("toilet_flush", "silent"),
    ),
    spec(
        category="plumbing_fixture",
        category_label="plumbing fixture",
        object_type="sink_with_tap",
        object_label="sink with tap",
        form_factor="pedestal_basin",
        form_label="pedestal washbasin",
        material="ceramic",
        material_label="glazed ceramic",
        body_color="white",
        color_label="gloss white",
        description=(
            "a single washbasin on one full-height pedestal with one integrated "
            "arched faucet; the faucet spout projects over the empty bowl and "
            "its circular aerator outlet is clearly visible"
        ),
        extra_negative="bathroom scene, vanity cabinet, mirror, loose plumbing, water stream, soap, multiple taps",
        height_cm=85,
        tolerance_cm=10,
        acoustic_profile_id="sink_with_tap_v1",
        default_event_class="water_tap_faucet",
        allowed_event_classes=("water_tap_faucet", "sink_filling_washing", "silent"),
    ),
    spec(
        category="plumbing_fixture",
        category_label="plumbing fixture",
        object_type="sink_with_tap",
        object_label="sink with tap",
        form_factor="counter_vanity",
        form_label="counter vanity basin",
        material="ceramic_and_wood",
        material_label="ceramic basin and painted wood vanity",
        body_color="white",
        color_label="white basin with light cabinet",
        description=(
            "one compact vanity cabinet supporting one inset ceramic basin and "
            "one integrated arched faucet; the spout mouth and aerator are fully "
            "visible above the empty bowl, with cabinet doors closed"
        ),
        extra_negative="bathroom scene, pedestal basin, mirror, loose plumbing, water stream, soap, double sink",
        height_cm=85,
        tolerance_cm=10,
        acoustic_profile_id="sink_with_tap_v1",
        default_event_class="water_tap_faucet",
        allowed_event_classes=("water_tap_faucet", "sink_filling_washing", "silent"),
    ),
    spec(
        category="plumbing_fixture",
        category_label="plumbing fixture",
        object_type="bathtub",
        object_label="bathtub",
        form_factor="freestanding",
        form_label="freestanding bathtub",
        material="acrylic",
        material_label="smooth acrylic",
        body_color="white",
        color_label="gloss white",
        description=(
            "a deep oval freestanding tub with a broad rim and one integrated "
            "deck-mounted filler spout at an end; the spout opening is clearly "
            "visible over the empty interior of the tub"
        ),
        extra_negative="bathroom scene, wall surround, shower curtain, person, water, separate floor faucet, accessories",
        height_cm=60,
        tolerance_cm=8,
        acoustic_profile_id="bathtub_v1",
        default_event_class="bathtub_filling_washing",
        allowed_event_classes=("bathtub_filling_washing", "silent"),
    ),
    spec(
        category="plumbing_fixture",
        category_label="plumbing fixture",
        object_type="bathtub",
        object_label="bathtub",
        form_factor="built_in_alcove",
        form_label="built-in alcove bathtub",
        material="acrylic",
        material_label="smooth acrylic",
        body_color="white",
        color_label="gloss white",
        description=(
            "a straight rectangular alcove tub body with one finished apron on "
            "the long front side and one compact rim-mounted filler spout at an "
            "end; the spout outlet is clearly visible over the empty tub"
        ),
        extra_negative="bathroom scene, tiled wall, shower curtain, freestanding oval tub, person, water, accessories",
        height_cm=55,
        tolerance_cm=8,
        acoustic_profile_id="bathtub_v1",
        default_event_class="bathtub_filling_washing",
        allowed_event_classes=("bathtub_filling_washing", "silent"),
    ),
    spec(
        category="plumbing_fixture",
        category_label="plumbing fixture",
        object_type="floor_drain",
        object_label="floor drain",
        form_factor="floor_drain",
        form_label="flush floor drain",
        material="steel",
        material_label="brushed stainless steel",
        body_color="silver",
        color_label="brushed silver",
        description=(
            "a low square drain body with a broad flat flange and one removable "
            "top grate; the grate perforations and dark drain throat directly "
            "beneath them are clearly visible from above"
        ),
        extra_negative="floor, bathroom scene, loose pipe, sink strainer, shower tray, water, debris, round manhole",
        height_cm=3,
        tolerance_cm=1,
        acoustic_profile_id="floor_drain_v1",
        default_event_class="drip",
        allowed_event_classes=("drip", "gurgling", "silent"),
    ),
    spec(
        category="plumbing_fixture",
        category_label="plumbing fixture",
        object_type="floor_drain",
        object_label="exposed drain trap",
        form_factor="exposed_p_trap",
        form_label="exposed P-trap assembly",
        material="brass",
        material_label="chrome-plated brass",
        body_color="silver",
        color_label="polished silver",
        description=(
            "one complete exposed P-shaped plumbing trap standing upright, with "
            "a short vertical inlet tailpiece and wall outlet; the circular open "
            "inlet at the top is clearly visible and all joints remain connected"
        ),
        extra_negative="sink, basin, wall, bathroom scene, disconnected pipes, multiple traps, water leak, floor drain grate",
        height_cm=25,
        tolerance_cm=5,
        acoustic_profile_id="floor_drain_v1",
        default_event_class="gurgling",
        allowed_event_classes=("drip", "gurgling", "silent"),
    ),
    spec(
        category="heating_fixture",
        category_label="architectural heating fixture",
        object_type="fireplace",
        object_label="fireplace",
        form_factor="masonry_open",
        form_label="open masonry fireplace",
        material="masonry",
        material_label="stone and brick masonry",
        body_color="warm_gray",
        color_label="warm gray stone",
        description=(
            "one self-contained masonry fireplace facade with a broad hearth, "
            "thick surround and one deep empty black firebox opening; the entire "
            "open firebox mouth is unobstructed and clearly visible"
        ),
        extra_negative="room wall, living room, burning fire, flames, logs, tools, mantel decorations, television",
        height_cm=110,
        tolerance_cm=15,
        acoustic_profile_id="fireplace_v1",
        default_event_class="fire",
        allowed_event_classes=("fire", "crackle", "silent"),
    ),
    spec(
        category="heating_fixture",
        category_label="architectural heating fixture",
        object_type="fireplace",
        object_label="wood stove",
        form_factor="wood_stove",
        form_label="freestanding wood-burning stove",
        material="cast_iron",
        material_label="cast iron",
        body_color="matte_black",
        color_label="matte black",
        description=(
            "a compact freestanding cast-iron stove on four short legs, with one "
            "closed glass firebox door on the front and a clearly visible row of "
            "adjustable combustion-air vent slots directly below the door"
        ),
        extra_negative="room scene, masonry fireplace, burning fire, flames, logs, chimney extending out of frame, tools",
        height_cm=75,
        tolerance_cm=10,
        acoustic_profile_id="fireplace_v1",
        default_event_class="crackle",
        allowed_event_classes=("fire", "crackle", "silent"),
    ),
)


def profile_payload(specification: dict[str, Any], reference: dict[str, Any]) -> dict:
    category = specification["category"]
    object_type = specification["object_type"]
    form_factor = specification["form_factor"]
    schema_id = f"{category}_{object_type}_{form_factor}_product_view_v1"
    contract = reference["generation_contract"]
    body_color = specification["body_color"]
    positive_template = (
        "One {body_color} {material} {form_factor} {object_type}, a household "
        "{category}: " + specification["description"] + "."
    )
    return {
        "schema": "avengine_attribute_profile_v1",
        "profile_schema_id": schema_id,
        "profile_revision": REVISION,
        "asset_class": "static_object",
        "lineage_group_id": "static_product_view_t2i_flux2_v1",
        "state_classification": "research_candidate",
        "taxonomy": {"category": category, "object_type": object_type},
        "base_template": {
            "template_id": "static_object_text_prompt_only_v1",
            "kind": "text_prompt_only",
            "artifact": None,
            "provenance_status": "verified",
            "usage_scope": "research_candidate",
        },
        "fixed_attributes": {
            "form_factor": form_factor,
            "material": specification["material"],
        },
        "sampled_attribute_domains": {"body_color": [body_color]},
        "forbidden_combinations": [],
        "generation_contract": {
            "route": "flux2_pixal3d_static_v1",
            "prompt_template_id": f"static_product_view_t2i_v1_{object_type}_{form_factor}",
            "base_acquisition_policy": {
                "policy_id": "static_object_per_request_one_shot_v1",
                "acquisition_unit": "one_frozen_asset_per_request",
                "sampled_domains_must_be_singleton": False,
                "downstream_instance_route": "flux2_pixal3d_static_v1",
                "profile_validation": "all_predeclared_requests_count_zero_hidden_failures",
            },
            "positive_template": positive_template,
            "pose_guard_prompt": contract["pose_guard_prompt"],
            "negative_prompt": (
                f"{contract['negative_prompt']}, {specification['extra_negative']}"
            ),
            "value_labels": {
                "category": {category: specification["category_label"]},
                "object_type": {object_type: specification["object_label"]},
                "form_factor": {form_factor: specification["form_label"]},
                "material": {
                    specification["material"]: specification["material_label"]
                },
                "body_color": {body_color: specification["color_label"]},
            },
            "model_revisions": dict(contract["model_revisions"]),
        },
        "target_physical_profiles": {
            "profile_id": f"{schema_id}_physical_candidate_v1",
            "control_attribute": None,
            "measurement": "height_cm",
            "mode": "absolute_measurement",
            "reference_value_cm": specification["height_cm"],
            "reference_provenance": {
                "status": "provisional",
                "source_id": f"{object_type}_{form_factor}_typical_retail_dimension_v1",
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
                    "target_value_cm": specification["height_cm"],
                    "tolerance_cm": specification["tolerance_cm"],
                }
            },
        },
        "rig_profile": None,
        "acoustic_profile": {
            "profile_id": specification["acoustic_profile_id"],
            "default_event_class": specification["default_event_class"],
            "allowed_event_classes": list(specification["allowed_event_classes"]),
            "selection_attributes": ["category", "object_type"],
        },
        "locked_attributes": ["category", "object_type", "form_factor", "material"],
        "qa_contract": {
            "subject_label": specification["object_label"],
            "attributes": {
                "body_color": {
                    "kind": "categorical",
                    "label": "body color or finish",
                    "value_labels": {body_color: specification["color_label"]},
                    "identification_question": (
                        "What is the body color or finish of {instance_label}?"
                    ),
                }
            },
        },
    }


def write_exact(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to replace differing profile: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--spear-root", required=True, type=Path)
    result.add_argument("--mirror-dir", required=True, type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    profile_dir = (
        args.spear_root
        / "data/controlled_source_attributes_v1/candidate_profiles/static_object"
    )
    reference_path = profile_dir / "audio_playback_floorstanding_speaker_product_view_v1.json"
    if not reference_path.is_file():
        raise SystemExit(f"validated audio_playback reference is missing: {reference_path}")

    sys.path.insert(0, str(args.spear_root))
    from tools import controlled_source_asset_schema as contracts

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    written: list[str] = []
    for existing_name in EXISTING_PROFILE_NAMES:
        source = profile_dir / existing_name
        if not source.is_file():
            raise SystemExit(f"existing profile is missing: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        contracts.validate_attribute_profile(payload)
        write_exact(args.mirror_dir / existing_name, source.read_bytes())
        written.append(f"reused {existing_name}")

    for specification in SPECS:
        payload = profile_payload(specification, reference)
        contracts.validate_attribute_profile(payload)
        combinations = contracts.legal_attribute_combinations(payload)
        if len(combinations) != 1:
            raise RuntimeError(
                f"{payload['profile_schema_id']} has {len(combinations)} combinations; expected one"
            )
        content = (
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
        ).encode("utf-8")
        name = f"{payload['profile_schema_id']}.json"
        write_exact(profile_dir / name, content)
        write_exact(args.mirror_dir / name, content)
        written.append(f"created {name}")

    for item in written:
        print(item)
    print(
        f"STATIC_SOUND_PROFILES_OK reused={len(EXISTING_PROFILE_NAMES)} "
        f"created={len(SPECS)} total_forms={len(EXISTING_PROFILE_NAMES) + len(SPECS)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
