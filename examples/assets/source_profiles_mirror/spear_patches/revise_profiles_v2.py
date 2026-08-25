#!/usr/bin/env python3
"""Two method revisions driven by measured 2D failures, with provenance.

bookshelf_speaker: the v1 method put three drivers on two of three baffles
against a declared two-way form factor.  television: the v1 method never shows
a sound outlet, and the static route deliberately forbids marking
emitter_feature_visible not_applicable, so the method has to put the outlet on
a visible face.

Neither is a seed retry: the same seed on the same profile would reproduce the
same wrong method, so both get a new profile revision and a forbidden-replay
record, following candidate_profile_revisions/animal/*/provenance.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPEAR = Path("/data/jzy/code/SPEAR-lead-b")
DATA = SPEAR / "data/controlled_source_attributes_v1"
PROFILE_DIR = DATA / "candidate_profiles/static_object"
REVISION_DIR = DATA / "candidate_profile_revisions/static_object"
WS = SPEAR / "tmp/audio_playback_statics_v1"

import sys

sys.path.insert(0, str(SPEAR))
from tools import controlled_source_asset_schema as contracts  # noqa: E402


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        contracts.canonical_json(payload).encode("utf-8")
    ).hexdigest()


def file_record(path: Path, profile: dict) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(SPEAR)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "profile_schema_id": profile["profile_schema_id"],
        "profile_revision": profile["profile_revision"],
        "canonical_sha256": canonical_sha256(profile),
    }


batch = json.loads((WS / "flux/flux2_batch_manifest.json").read_text(encoding="utf-8"))
requests = json.loads(
    (WS / "inputs/instance_requests.json").read_text(encoding="utf-8")
)
request_by_instance = {
    item["instance_id"]: item for item in requests["requests"]
}
candidate_by_instance = {item["instance_id"]: item for item in batch["candidates"]}


REVISIONS = {
    "audio_playback_bookshelf_speaker_product_view_v1": {
        "profile_revision": "2026_08_26_v2_two_way_driver_count_method_revision",
        "prompt_template_id": "static_product_view_t2i_v2_two_way_driver_count",
        "positive_template": (
            "One {finish} {material} {form_factor} {object_type}, a household "
            "{category}: a compact upright rectangular loudspeaker box, clearly "
            "taller than it is wide, standing on its own flat plinth. Its front "
            "baffle carries exactly two round drivers and nothing else: one "
            "large woofer cone in the lower half, and one small dome tweeter "
            "above it. There is no third driver, no midrange cone and no "
            "opening between them. A single circular port sits below the "
            "woofer. The side, top and back panels are plain and flat with no "
            "controls and no cloth."
        ),
        "extra_negative": (
            "three drivers, third driver, midrange driver, midrange cone, "
            "three-way speaker, four drivers, driver array, stacked drivers"
        ),
        "kind": "driver_count_method_revision",
        "reason": (
            "The v1 method declares the fixed attribute "
            "form_factor=sealed_two_way_cabinet but states the driver count "
            "only positively. Two of its three finishes came back as three-way "
            "cabinets with a midrange cone between the tweeter and the woofer. "
            "Those images are good loudspeakers, so a seed retry would keep "
            "drawing from the same under-constrained method and would keep "
            "producing records that claim a two-way form factor the mesh does "
            "not have. The v2 method states the driver count as an exclusion "
            "in the positive template and guards the third driver in the "
            "negative prompt."
        ),
        "failed_instances": [
            "audio_playback_bookshelf_speaker_product_view_ae7e486cdf1d",
            "audio_playback_bookshelf_speaker_product_view_842baebdba9a",
        ],
    },
    "audio_playback_television_product_view_v1": {
        "profile_revision": "2026_08_26_v2_visible_bottom_grille_method_revision",
        "prompt_template_id": "static_product_view_t2i_v2_visible_bottom_grille",
        "positive_template": (
            "One {material} {form_factor} {object_type} standing on a "
            "{stand_type}, a household {category}: one large living-room "
            "flat-screen television far wider than it is tall, with a "
            "completely blank matte dark screen showing nothing at all. Across "
            "the full width of the bottom bezel, directly under the screen, "
            "runs one narrow horizontal perforated speaker grille strip, "
            "clearly visible from the front. The bezel is otherwise plain and "
            "the back is flat. It rests on a table, not on a wall."
        ),
        "extra_negative": (
            "computer monitor, desktop monitor, thin column stand, "
            "office desk, keyboard, mouse, bezel with no grille, "
            "hidden speakers, small screen"
        ),
        "kind": "visible_emitter_feature_method_revision",
        "reason": (
            "Every v1 television candidate had to be rejected on the "
            "emitter_feature_visible hard gate: modern television speakers fire "
            "downward behind an invisible bottom vent, and the static review "
            "route deliberately refuses not_applicable on that gate, so no "
            "seed can rescue the method. Placing an emitter anchor on a "
            "surface no reviewer can see is exactly what the gate exists to "
            "prevent. The v2 method asks for the front-facing bottom grille "
            "strip that mid-range televisions actually have, which makes the "
            "anchor reviewable. The same revision also pushes the pedestal "
            "value away from computer monitors, which is what the v1 pedestal "
            "candidate came back as."
        ),
        "failed_instances": [
            "audio_playback_television_product_view_fe83ed5e6682",
            "audio_playback_television_product_view_5daa484aeaaf",
            "audio_playback_television_product_view_5f95ce65d7a7",
        ],
    },
}

# The pedestal value label pushed the generator toward monitor stands.
LABEL_OVERRIDES = {
    "audio_playback_television_product_view_v1": {
        "stand_type": {
            "central_pedestal": "wide flat plate pedestal",
            "two_splayed_feet": "two splayed feet at the outer edges",
        }
    }
}

REVISION_DIR.mkdir(parents=True, exist_ok=True)
for schema_id, revision in REVISIONS.items():
    path = PROFILE_DIR / f"{schema_id}.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    old_record = file_record(path, old)

    new = json.loads(json.dumps(old))
    new["profile_revision"] = revision["profile_revision"]
    contract = new["generation_contract"]
    contract["prompt_template_id"] = revision["prompt_template_id"]
    contract["positive_template"] = revision["positive_template"]
    contract["negative_prompt"] = (
        f"{contract['negative_prompt']}, {revision['extra_negative']}"
    )
    for attribute, labels in LABEL_OVERRIDES.get(schema_id, {}).items():
        contract["value_labels"][attribute] = labels
        new["qa_contract"]["attributes"][attribute]["value_labels"] = labels
    contracts.validate_attribute_profile(new)
    path.write_text(
        json.dumps(new, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    triggering = []
    for instance_id in revision["failed_instances"]:
        request = request_by_instance[instance_id]
        candidate = candidate_by_instance[instance_id]
        triggering.append(
            {
                "instance_id": instance_id,
                "request_sha256": request["request_sha256"],
                "candidate_sha256": candidate["candidate"]["sha256"],
                "sampled_attributes": candidate["sampled_attributes"],
            }
        )

    provenance = {
        "schema": "avengine_controlled_profile_method_revision_provenance_v1",
        "profile": file_record(path, new),
        "method_revision": {
            "kind": revision["kind"],
            "supersedes_profile_revision": old["profile_revision"],
            "superseded_profile_canonical_sha256": old_record["canonical_sha256"],
            "reason": revision["reason"],
            "this_is_not_a_seed_retry": True,
            "same_candidate_retry_allowed": False,
            "same_generation_seed_retry_allowed": False,
            "candidate_ranking_allowed": False,
            "next_execution_requires_new_profile_bound_request": True,
            "next_execution_requires_new_batch_seed": True,
            "forbidden_replay_request_sha256": [
                item["request_sha256"] for item in triggering
            ],
            "forbidden_replay_candidate_sha256": [
                item["candidate_sha256"] for item in triggering
            ],
        },
        "triggering_failure": {
            "review_batch": str(
                (WS / "review_2d/review_batch_manifest.json").relative_to(SPEAR)
            ),
            "reviewer": (
                "first_author_claude_visual_review_under_project_lead_"
                "delegation_20260826"
            ),
            "instances": triggering,
        },
    }
    out = REVISION_DIR / revision["profile_revision"] / "provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"revised {schema_id} -> {revision['profile_revision']}")
    print(f"  provenance {out.relative_to(SPEAR)}")
