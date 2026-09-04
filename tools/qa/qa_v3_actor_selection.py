"""Actor selection documents for QA v3 batches (neutral home for the helpers).

These helpers used to live in design_qa_v3_pilot_batch.py, the historical
fixed-camera assembler.  The current room-centric generators (scene batch,
extended profiles, n-actor planner) import them from here so that the
historical CLI can be retired without dragging its production semantics
along.  Behaviour is unchanged: given a registry record and the content
snapshot, resolve the Blueprint, graph-derived skeletal mesh and animation
packages to their physical authorised sources.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone


def _mesh_package_for(asset, snap):
    su = asset["runtime_backends"]["spear_unreal"]
    mesh_dir_pkg = su["idle_animation"].split(".", 1)[0].rsplit("/", 1)[0]
    gate = mesh_dir_pkg.rsplit("/", 1)[-1]
    phys_dir = os.path.join(snap, "MyAssets/Audioset/Meshes", gate)
    names = [f[:-7] for f in os.listdir(phys_dir) if f.endswith(".uasset")]
    for n in names:
        if n + "_Skeleton" in names:
            return mesh_dir_pkg + "/" + n
    if "runtime" in names:
        return mesh_dir_pkg + "/runtime"
    raise RuntimeError(f"cannot identify skeletal mesh in {phys_dir}: {names}")


def _actor_entry(slot, asset_id, by_id, snap):
    rec = by_id[asset_id]
    su = rec["runtime_backends"]["spear_unreal"]
    bp = su["blueprint_class_path"]
    bp_pkg = bp.split(".", 1)[0]
    mesh_pkg = _mesh_package_for(rec, snap)
    mesh_name = mesh_pkg.rsplit("/", 1)[-1]

    def phys(package):
        p = os.path.join(snap, package.split("/Game/", 1)[1] + ".uasset")
        if not os.path.isfile(p):
            raise RuntimeError(f"missing physical source: {p}")
        return p

    return {
        "asset_id": asset_id,
        "legacy_timeline_actor_id": f"{rec['identity']['species_id']}_{slot[-1]}",
        "physical_authorized_internal_sources": {
            "blueprint": phys(bp_pkg),
            "graph_derived_mesh": phys(mesh_pkg),
            "idle": phys(su["idle_animation"].split(".", 1)[0]),
            "walking": phys(su["walking_animation"].split(".", 1)[0]),
        },
        "profile_alias": asset_id,
        "revision": rec["revision"],
        "source_slot_id": slot,
        "ue_binding": {
            "blueprint_object_path": bp,
            "blueprint_package": bp_pkg,
            "graph_derived_mesh": {
                "derivation": "direct graph dependency of the selected Blueprint; profile binds blueprint_component and declares no standalone mesh path",
                "object_path": f"{mesh_pkg}.{mesh_name}",
                "package": mesh_pkg,
            },
            "idle_object_path": su["idle_animation"],
            "idle_package": su["idle_animation"].split(".", 1)[0],
            "profile_skeletal_mesh_binding": su["skeletal_mesh_binding"],
            "profile_skeletal_mesh_path": su["skeletal_mesh_path"],
            "walking_object_path": su["walking_animation"],
            "walking_package": su["walking_animation"].split(".", 1)[0],
        },
    }


def _selection_doc(a1, a2, by_id, snap):
    return {
        "schema": "avengine_apartment_actor_selection_v1",
        "asset_authorization": "verified_internal",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": "QA v3 dual-source pilot batch; research only.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "actors": [_actor_entry("source1", a1, by_id, snap),
                   _actor_entry("source2", a2, by_id, snap)],
    }
