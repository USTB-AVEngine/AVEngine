#!/usr/bin/env python3
"""Revise four static plumbing profiles after repeat failures across two seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


OLD_REVISION = "2026_08_26_v1_static_sound_forms"

REVISIONS = {
    "plumbing_fixture_bathtub_freestanding_product_view_v1": {
        "old_sha": "3479f4695cfba534a2bee2fdd0071502eb07ba64023361448cf85d53b71964de",
        "revision": "2026_08_26_v2_integrated_filler_no_external_plumbing_method_revision",
        "prompt_id": "static_product_view_t2i_v2_freestanding_tub_integrated_filler",
        "positive": (
            "One {body_color} {material} {form_factor} {object_type}, a household "
            "{category}: one deep oval freestanding tub with a broad continuous rim "
            "and a plain fully enclosed outer shell. At one end, one short arched "
            "deck-mounted filler spout is physically fused into the tub rim; its "
            "single circular water outlet is clearly visible over the empty tub. "
            "The underside and exterior are smooth and contain no visible pipes, "
            "hoses, traps, valves or plumbing connectors."
        ),
        "negative": (
            "missing faucet, tub without filler, faucet not attached to rim, "
            "separate floor faucet, freestanding floor tap, external pipe, exposed "
            "plumbing, pipe under tub, hanging drain pipe, bulbous connector, hose"
        ),
        "kind": "integrated_visible_filler_method_revision",
        "reason": (
            "The v1 method failed in two different ways across two frozen requests. "
            "The first candidate contained the requested filler in 2D but Pixal3D "
            "grew a long exposed external pipe and bulbous plumbing below the tub, "
            "causing a 3D construction rejection. The second seed produced a clean "
            "tub but omitted the filler entirely, so emitter_feature_visible failed. "
            "The method must bind one compact filler to the rim and explicitly keep "
            "all underside plumbing enclosed; another blind seed retry is not enough."
        ),
        "failures": [
            ("plumbing_fixture_bathtub_freestanding_product_view_cb6dabe11eab", "cb6dabe11eab71be06854910a3cb6241b6c8e06a5719b00307bc3ab6c0472378", "894ca4fc76c9fc45d2f816baaf6d0a2a033b8043a1925362fcb111dc2ccf0bb5"),
            ("plumbing_fixture_bathtub_freestanding_product_view_58b4917701d7", "58b4917701d7134b4c142365d56b16b250cb5d3ecda61f2ee446c7567d2e4e9f", "2643bfec1146eb50bca0f3c00d363f0667ad909421bce6e2d427b6538a682ecc"),
        ],
    },
    "plumbing_fixture_floor_drain_exposed_p_trap_product_view_v1": {
        "old_sha": "6ecff6e16b977281484f25acfdd061655ad192cc9a69a06c57a6b165223a27d0",
        "revision": "2026_08_26_v2_true_u_bend_water_seal_method_revision",
        "prompt_id": "static_product_view_t2i_v2_exposed_p_trap_u_bend",
        "positive": (
            "One {body_color} {material} {form_factor} {object_type}, a household "
            "{category}: one continuous chrome plumbing P-trap assembly whose "
            "silhouette unmistakably contains a deep U-shaped 180-degree water-seal "
            "bend. One vertical inlet descends into the left side of the U, the pipe "
            "rises on the right side, then turns once into one horizontal wall outlet. "
            "There are exactly two open ends and no branch intersection. The circular "
            "top inlet is clearly visible."
        ),
        "negative": (
            "straight vertical pipe, straight-through pipe, tee fitting, T junction, "
            "Y junction, perpendicular side branch, three openings, cross fitting, "
            "missing U bend, shallow elbow, disconnected pipe"
        ),
        "kind": "true_p_trap_geometry_method_revision",
        "reason": (
            "Two different v1 requests both produced the same wrong object: a straight "
            "vertical tube with a perpendicular side branch and no U-shaped water seal. "
            "That repeat failure shows the term exposed P-trap was under-specified for "
            "the image method. The v2 method defines the complete two-ended flow path "
            "and excludes tee and three-opening fittings."
        ),
        "failures": [
            ("plumbing_fixture_floor_drain_exposed_p_trap_product_view_e4f7dfc371c3", "e4f7dfc371c3214ff78b9b7f6087614079f1ba6f44e88ceb5f2dc2afe177d494", "a85092c9b6fba8dafbecd08c2b624f0fac9decd20c7d0474051c11e13cdb23f8"),
            ("plumbing_fixture_floor_drain_exposed_p_trap_product_view_ba6c9a2c9253", "ba6c9a2c9253700371593c405ff16769ad2332d5bf1aab09af9a0457914f3397", "7a8d87a1582225a71d6b61eb98fc2137c9efa6d1c3b34995df2e433f6211af22"),
        ],
    },
    "plumbing_fixture_toilet_floor_close_coupled_product_view_v1": {
        "old_sha": "1ebe519dbf19b259836193847a6358225ef8ed91c733fcde23da9f2cd1a7dcef",
        "revision": "2026_08_26_v2_visible_flush_jet_ring_method_revision",
        "prompt_id": "static_product_view_t2i_v2_close_coupled_visible_flush_jets",
        "positive": (
            "One {body_color} {material} {form_factor} {object_type}, a household "
            "{category}: one floor-standing close-coupled toilet with one integrated "
            "rear cistern. The seat and lid are raised together. Use a high three-quarter "
            "view into an empty bowl so a continuous dark flush-water slot, or a row of "
            "distinct round flush jet holes, is unmistakably visible directly beneath "
            "the inner rear rim. The outlet feature must not be hidden by water or shadow."
        ),
        "negative": (
            "smooth featureless inner rim, hidden flush outlet, concealed rim, rimless "
            "bowl without visible jets, water obscuring rim, closed lid, lowered seat, "
            "wall-hung bowl, detached cistern"
        ),
        "kind": "visible_flush_outlet_method_revision",
        "reason": (
            "Both v1 seeds produced acceptable close-coupled toilet forms, but neither "
            "showed a flush slot or jet holes under the rim. Since the static 2D route "
            "forbids emitter_feature_visible=not_applicable, the method itself must use "
            "a higher view and explicitly expose a reviewable water outlet."
        ),
        "failures": [
            ("plumbing_fixture_toilet_floor_close_coupled_product_view_c5a49e583b82", "c5a49e583b82c651831671133febed2bb731f2b5bd7c4a4964b8068546a8f85e", "a34c92b20031505151856b022aba1b6d57b94d31d7ae0405c572cc018230c6fd"),
            ("plumbing_fixture_toilet_floor_close_coupled_product_view_002c2d2486cd", "002c2d2486cdb1cf6f542ec923c2002134a21143e77bfcbd37355cad491e7927", "c2f21f4fe47d7bc5328ac815526fd3d1e0e72dbab1aa99e6fbffc8f8c79e45b7"),
        ],
    },
    "plumbing_fixture_toilet_wall_hung_product_view_v1": {
        "old_sha": "62d28fdf1222e91aceb58a458aa19ae7373baf8e293ea86cb9a29c7e3331de94",
        "revision": "2026_08_26_v2_cisternless_wall_bowl_method_revision",
        "prompt_id": "static_product_view_t2i_v2_wall_hung_cisternless_visible_jets",
        "positive": (
            "One {body_color} {material} {form_factor} {object_type}, a household "
            "{category}: one compact cisternless wall-hung toilet bowl in its installed "
            "orientation. The flat vertical rear mounting face is clearly visible and "
            "the bowl projects horizontally from it without any floor pedestal, foot, "
            "base, tank or cistern. The seat and lid are raised. A row of distinct flush "
            "jet holes beneath the inner rear rim is clearly visible from above."
        ),
        "negative": (
            "floor-standing toilet, floor pedestal, foot, base touching floor, exposed "
            "cistern, tank, close-coupled toilet, detached tank, hidden flush outlet, "
            "smooth featureless inner rim, closed lid, lowered seat"
        ),
        "kind": "wall_hung_form_and_visible_flush_method_revision",
        "reason": (
            "Two v1 seeds both collapsed the wall-hung form into a floor-standing "
            "close-coupled toilet with a large cistern, and neither exposed a flush outlet. "
            "The shared rest-on-floor pose guard reinforced the wrong form. The v2 method "
            "uses an installed wall-mount presentation, bans every floor support and "
            "cistern, and exposes the rear-rim jets for emitter review."
        ),
        "wall_mount_pose": True,
        "failures": [
            ("plumbing_fixture_toilet_wall_hung_product_view_dc3d760ab8b8", "dc3d760ab8b86df1f1ce91d0a7c4a987aa98aac89188d0eb56d8669f9fac9217", "edb827d84a0f82c07e4943425920361af945fa761df606b08947ec7976b102a6"),
            ("plumbing_fixture_toilet_wall_hung_product_view_7b43d6aabbc5", "7b43d6aabbc5716b5d2046313dafa42e4cd98a4360ca10b6bef1327868200bcb", "2aaed794742ccc9278300972f4ab86ea521b4fcbcba15c73676278a530dc7be8"),
        ],
    },
}


def canonical_sha256(payload: dict, contracts) -> str:
    return hashlib.sha256(contracts.canonical_json(payload).encode("utf-8")).hexdigest()


def file_record(path: Path, profile: dict, root: Path, contracts) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "profile_schema_id": profile["profile_schema_id"],
        "profile_revision": profile["profile_revision"],
        "canonical_sha256": canonical_sha256(profile, contracts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spear-root", required=True, type=Path)
    parser.add_argument("--avengine-root", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.spear_root))
    from tools import controlled_source_asset_schema as contracts

    spear_profiles = args.spear_root / "data/controlled_source_attributes_v1/candidate_profiles/static_object"
    mirror_profiles = args.avengine_root / "examples/assets/source_profiles_mirror/candidate_profiles/static_object"
    prepared = []
    for schema_id, revision in REVISIONS.items():
        spear_path = spear_profiles / f"{schema_id}.json"
        mirror_path = mirror_profiles / f"{schema_id}.json"
        if spear_path.read_bytes() != mirror_path.read_bytes():
            raise RuntimeError(f"{schema_id}: SPEAR and AVEngine profiles diverged")
        old = json.loads(spear_path.read_text(encoding="utf-8"))
        if old["profile_revision"] != OLD_REVISION:
            raise RuntimeError(f"{schema_id}: unexpected old revision {old['profile_revision']}")
        if canonical_sha256(old, contracts) != revision["old_sha"]:
            raise RuntimeError(f"{schema_id}: old canonical identity changed")

        new = json.loads(json.dumps(old))
        new["profile_revision"] = revision["revision"]
        contract = new["generation_contract"]
        contract["prompt_template_id"] = revision["prompt_id"]
        contract["positive_template"] = revision["positive"]
        contract["negative_prompt"] = f"{contract['negative_prompt']}, {revision['negative']}"
        if revision.get("wall_mount_pose"):
            old_phrase = "resting stably on a flat neutral surface"
            new_phrase = (
                "presented in its installed wall-mounted orientation with the flat rear "
                "mounting face vertical and the bowl projecting horizontally, without "
                "showing a wall or bracket"
            )
            if contract["pose_guard_prompt"].count(old_phrase) != 1:
                raise RuntimeError("wall-hung pose guard did not match v1")
            contract["pose_guard_prompt"] = contract["pose_guard_prompt"].replace(old_phrase, new_phrase, 1)
            contract["negative_prompt"] = contract["negative_prompt"].replace("floating object, ", "")
        contracts.validate_attribute_profile(new)
        for root in (
            args.spear_root / "data/controlled_source_attributes_v1/candidate_profile_revisions/static_object",
            args.avengine_root / "examples/assets/source_profiles_mirror/candidate_profile_revisions/static_object",
        ):
            path = root / revision["revision"] / "provenance.json"
            if path.exists():
                raise RuntimeError(f"refusing to replace method provenance: {path}")
        prepared.append((schema_id, revision, spear_path, mirror_path, old, new))

    for schema_id, revision, spear_path, mirror_path, old, new in prepared:
        content = json.dumps(new, ensure_ascii=False, indent=1) + "\n"
        spear_path.write_text(content, encoding="utf-8")
        mirror_path.write_text(content, encoding="utf-8")
        failures = [
            {
                "instance_id": instance_id,
                "request_sha256": request_sha,
                "candidate_sha256": candidate_sha,
                "sampled_attributes": {"body_color": "white" if "toilet" in schema_id or "bathtub" in schema_id else "silver"},
            }
            for instance_id, request_sha, candidate_sha in revision["failures"]
        ]
        provenance = {
            "schema": "avengine_controlled_profile_method_revision_provenance_v1",
            "profile": file_record(spear_path, new, args.spear_root, contracts),
            "method_revision": {
                "kind": revision["kind"],
                "supersedes_profile_revision": OLD_REVISION,
                "superseded_profile_canonical_sha256": revision["old_sha"],
                "reason": revision["reason"],
                "this_is_not_a_seed_retry": True,
                "same_candidate_retry_allowed": False,
                "same_generation_seed_retry_allowed": False,
                "candidate_ranking_allowed": False,
                "next_execution_requires_new_profile_bound_request": True,
                "next_execution_requires_new_batch_seed": True,
                "forbidden_replay_request_sha256": [item["request_sha256"] for item in failures],
                "forbidden_replay_candidate_sha256": [item["candidate_sha256"] for item in failures],
            },
            "triggering_failure": {
                "review_batch": "tmp/static_sound_sources_retry_20260826_r2/review_2d/review_batch_manifest.json",
                "reviewer": "codex_full_resolution_visual_review_under_owner_instruction_20260826",
                "instances": failures,
            },
        }
        for root in (
            args.spear_root / "data/controlled_source_attributes_v1/candidate_profile_revisions/static_object",
            args.avengine_root / "examples/assets/source_profiles_mirror/candidate_profile_revisions/static_object",
        ):
            path = root / revision["revision"] / "provenance.json"
            if path.exists():
                raise RuntimeError(f"refusing to replace method provenance: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(provenance, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"revised {schema_id} -> {revision['revision']}")

    print(f"STATIC_PLUMBING_METHOD_REVISIONS_OK profiles={len(prepared)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
