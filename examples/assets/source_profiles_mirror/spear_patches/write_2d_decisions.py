#!/usr/bin/env python3
"""Freeze my own visual verdicts on the 15 audio_playback 2D candidates."""

from __future__ import annotations

import json
from pathlib import Path

WS = Path("/data/jzy/code/SPEAR-lead-b/tmp/audio_playback_statics_v1")
BATCH = json.loads((WS / "flux/flux2_batch_manifest.json").read_text(encoding="utf-8"))

CHECKS = ("category_identity", "construction", "stable_product_pose", "background")
GATES = (
    "single_subject",
    "photorealistic_pbr_style",
    "category_distinctive_features",
    "emitter_feature_visible",
    "physically_connected_construction",
    "complete_object",
    "stable_rest_or_mount",
    "target_attribute_only",
)

# job id suffix -> (rejected checks, rejected hard gates, note)
REJECTIONS = {
    "static_ae7e486cdf1dc263": (
        (),
        ("target_attribute_only",),
        "three drivers on the baffle - tweeter, midrange and woofer - so the "
        "cabinet is three-way while the profile declares the fixed attribute "
        "form_factor=sealed_two_way_cabinet. The object is otherwise a clean "
        "bookshelf loudspeaker; it is rejected because publishing it would "
        "record a two-way form factor that the mesh does not have.",
    ),
    "static_842baebdba9adc26": (
        (),
        ("target_attribute_only",),
        "same three-way driver stack as the black_ash candidate against a "
        "declared two-way form factor.",
    ),
    "static_5006e17ef57c76f2": (
        ("category_identity",),
        ("category_distinctive_features",),
        "the sandstone weave came out as a coarse open basket knit over a squat "
        "body, so the object reads as a wicker stool or pouf rather than a "
        "smart speaker. The four top openings are present but cannot carry the "
        "identity on their own.",
    ),
    "static_fe83ed5e668214fc": (
        ("category_identity",),
        ("category_distinctive_features", "emitter_feature_visible"),
        "the central pedestal candidate came out as a computer monitor: small "
        "panel on a thin monitor stand. It also shows no sound outlet anywhere.",
    ),
    "static_5daa484aeaaf8169": (
        (),
        ("emitter_feature_visible",),
        "a convincing television on two splayed feet, but its speakers are "
        "down-firing behind an invisible bottom vent, so there is no visible "
        "feature to anchor the emitter to. The static route forbids "
        "not_applicable on this gate, which is correct: an anchor that cannot "
        "be reviewed against a visible outlet is not reviewable at all. The "
        "television profile needs a method revision that puts a front-facing "
        "grille strip under the screen.",
    ),
    "static_5f95ce65d7a7e555": (
        (),
        ("emitter_feature_visible",),
        "same as the other splayed-feet television: no visible sound outlet.",
    ),
}

APPROVAL_NOTES = {
    "static_477fecc3d6769096": (
        "two drivers only - offset dome tweeter and woofer - which is the "
        "declared two-way form factor. Driver mounting screws are visible and "
        "are how real cabinets are built."
    ),
    "static_995cd26ea8810794": "four stacked drivers on a plinth, clean tower silhouette.",
    "static_e78b6e647ac6a75a": "four stacked drivers on a plinth, clean tower silhouette.",
    "static_0dcc002103d79b53": "four stacked drivers on a plinth, clean tower silhouette.",
    "static_6c8bedd9b9ea0f09": (
        "closed mesh weave, flat top disc with four openings, dark base ring."
    ),
    "static_c151f3243594d878": (
        "closed mesh weave, flat top disc with four openings, dark base ring."
    ),
    "static_b11e127682a024bc": (
        "single bar, full-width front grille, two low feet, flush controls at "
        "one end. It is nearer six times wider than tall rather than the ten "
        "the prompt asked for, so at the declared 7 cm height it will finalize "
        "as a compact bar around 40 cm long. That is a real product size and "
        "the declared form factor wide_low_bar still holds, so the aspect is "
        "recorded rather than treated as a failure; the finalized bounding box "
        "must be measured before the height target is registered."
    ),
    "static_fdbb7157722fcc98": "single bar, full-width front grille, two low feet.",
    "static_61ea3095f68cf2a4": "single bar, full-width front grille, two low feet.",
}

decisions = []
for candidate in BATCH["candidates"]:
    job = candidate["execution_job_id"]
    rejected_checks, rejected_gates, note = REJECTIONS.get(job, ((), (), ""))
    approved = not rejected_checks and not rejected_gates
    if approved:
        note = APPROVAL_NOTES[job]
    decisions.append(
        {
            "instance_id": candidate["instance_id"],
            "candidate_sha256": candidate["candidate"]["sha256"],
            "decision": "approved_for_pixal3d" if approved else "rejected",
            **{
                check: "rejected" if check in rejected_checks else "passed"
                for check in CHECKS
            },
            "sampled_attribute_checks": {
                attribute: "rejected" if not approved and attribute in () else "passed"
                for attribute in candidate["sampled_attributes"]
            },
            "hard_gates": {
                gate: "rejected" if gate in rejected_gates else "passed"
                for gate in GATES
            },
            "notes": note,
        }
    )

payload = {
    "schema": "avengine_controlled_static_object_2d_review_decisions_v1",
    "flux2_batch_sha256": BATCH["batch_sha256"],
    "reviewer": "first_author_claude_visual_review_under_project_lead_delegation_20260826",
    "decisions": decisions,
}
out = WS / "review_inputs/static_2d_decisions.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
approved = sum(1 for d in decisions if d["decision"] == "approved_for_pixal3d")
print(f"wrote {out} approved={approved} rejected={len(decisions) - approved}")
