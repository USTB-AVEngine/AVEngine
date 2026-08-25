#!/usr/bin/env python3
"""Freeze the static-object visual decisions for the nine reconstructions.

Two measurements decided these, not the renders alone.  The five-view renders
cull backfaces, so a closed box seen from behind looks open - every mesh here
measured 0 to 30 boundary edges out of about 1.5 M, so none of them is open.
What the renders do show honestly is debris and lean, and both were measured:
debris as the surface area outside the largest connected component, lean as
the elevation of the area-weighted long principal axis.
"""

from __future__ import annotations

import json
from pathlib import Path

WS = Path("/data/jzy/code/SPEAR-lead-b/tmp/audio_playback_statics_v1")
BATCH = json.loads(
    (WS / "static_review/static_object_review_batch_manifest.json").read_text(
        encoding="utf-8"
    )
)

CHECKS = (
    "silhouette_and_category_identity",
    "emitter_feature_visible",
    "material_and_declared_attributes",
    "physically_plausible_construction",
    "no_disconnected_or_floating_parts",
)

# suffix -> (failed checks, caveats, notes)
VERDICTS = {
    "477fecc3d676": (
        (),
        [
            "long_principal_axis_elevation_78_3_deg_so_the_cabinet_leans_11_7_deg_backward",
            "no_stage_in_the_static_chain_levels_pitch_or_roll",
        ],
        "Cleanest reconstruction of the nine: 6 components, 0.075 percent of "
        "surface area outside the main shell, 0 boundary edges. Both drivers "
        "and the port survive on the baffle. It leans 11.7 degrees backward, "
        "which the finalizer cannot correct because it applies yaw only.",
    ),
    "61ea3095f68c": (
        (),
        ["long_principal_axis_elevation_10_5_deg_so_the_bar_rolls_10_5_deg"],
        "Clean bar: full-width front grille, two low feet, 1.59 percent debris "
        "area, 0 boundary edges. Rolled 10.5 degrees.",
    ),
    "b11e127682a0": (
        (),
        ["long_principal_axis_elevation_8_6_deg_so_the_bar_rolls_8_6_deg"],
        "Clean bar, 7 components and 1.47 percent debris area, the lowest "
        "component count of the batch. Rolled 8.6 degrees.",
    ),
    "fdbb7157722f": (
        (),
        [
            "long_principal_axis_elevation_9_2_deg_so_the_bar_rolls_9_2_deg",
            "769_components_but_only_1_47_percent_of_area_outside_the_main_shell",
        ],
        "The perforated grille reads clearly as a grille, which is what the "
        "emitter anchor is placed on. The high component count is the "
        "perforation pattern shedding specks, not a broken shell: 1.47 percent "
        "of area sits outside the main component and no debris is visible in "
        "any of the five views.",
    ),
    "0dcc002103d7": (
        ("no_disconnected_or_floating_parts", "physically_plausible_construction"),
        [],
        "5.88 percent of surface area sits outside the main shell, and the "
        "front and quarter views show a separate hallucinated object - a "
        "bottle or finial with two rods - resting on an extended base slab. "
        "It also leans 22.5 degrees, the worst of the batch.",
    ),
    "995cd26ea881": (
        ("no_disconnected_or_floating_parts",),
        ["least_leaning_mesh_of_the_batch_at_5_7_deg"],
        "The cabinet itself is solid and the least tilted of all nine, but "
        "3.03 percent of area is dangling wire and rod geometry at the base, "
        "visible in the side and quarter views. Worth regenerating rather than "
        "discarding: the cabinet is good.",
    ),
    "e78b6e647ac6": (
        ("physically_plausible_construction",),
        [],
        "The cabinet is reconstructed split open: the front baffle stands "
        "detached from the side panels, so the object reads as a flat pack "
        "mid-assembly rather than a loudspeaker. 10013 non-manifold edges, the "
        "worst of the batch.",
    ),
    "6c8bedd9b9ea": (
        ("no_disconnected_or_floating_parts",),
        [],
        "33.72 percent of surface area lies outside the main shell - a third "
        "of the mesh is not attached to the speaker. It also leans 15.4 "
        "degrees.",
    ),
    "c151f3243594": (
        ("physically_plausible_construction",),
        ["debris_area_only_1_08_percent_so_the_shell_itself_is_clean"],
        "The shell is clean but the cylinder is rolled 18.9 degrees off "
        "vertical: it reads as tipping over rather than standing. A cylinder "
        "that cannot rest on its own base is not a plausible resting pose, and "
        "nothing downstream corrects roll.",
    ),
}

decisions = []
for entry in BATCH["reviews"]:
    instance_id = entry["instance_id"]
    suffix = instance_id.rsplit("_", 1)[-1]
    failed, caveats, notes = VERDICTS[suffix]
    approved = not failed
    review = json.loads(
        (WS / "static_review" / entry["review"]["path"]).read_text(encoding="utf-8")
    )
    decisions.append(
        {
            "instance_id": instance_id,
            "review_sha256": entry["review_sha256"],
            "decision": (
                "approved_for_watertight_finalization" if approved else "rejected"
            ),
            "checks": {check: check not in failed for check in CHECKS},
            "attribute_evidence": {
                attribute: (
                    "passed_raw_pbr_visual"
                    if approved
                    else "not_visually_assessable"
                )
                for attribute in review["sampled_attributes"]
            },
            "caveats": caveats,
            "notes": notes,
        }
    )

payload = {
    "schema": "avengine_controlled_static_object_review_decisions_v1",
    "static_object_review_batch_sha256": BATCH["review_batch_sha256"],
    "decisions": decisions,
}
out = WS / "review_inputs/static_3d_decisions.json"
out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
approved = sum(1 for d in decisions if d["decision"].startswith("approved"))
print(f"wrote {out} approved={approved} rejected={len(decisions) - approved}")
