"""Resolve selected articulated or rigid source assets to their UE content.

Current QA generators share this helper. Explicit registry mesh paths take
precedence; legacy articulated profiles may use one unambiguous mesh beside
their animation assets. Actual capture verifies the spawned mesh binding.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone


def _mesh_package_for(asset, snap):
    su = asset["runtime_backends"]["spear_unreal"]
    explicit = su.get("skeletal_mesh_path")
    if explicit:
        return explicit.split(".", 1)[0]
    mesh_dir_pkg = su["idle_animation"].split(".", 1)[0].rsplit("/", 1)[0]
    if not mesh_dir_pkg.startswith("/Game/"):
        raise RuntimeError(f"unsupported skeletal mesh package directory: {mesh_dir_pkg}")
    phys_dir = os.path.join(snap, mesh_dir_pkg.removeprefix("/Game/"))
    names = sorted(f[:-7] for f in os.listdir(phys_dir) if f.endswith(".uasset"))
    candidates = [n for n in names if n + "_Skeleton" in names]
    if len(candidates) == 1:
        return mesh_dir_pkg + "/" + candidates[0]
    if not candidates and "runtime" in names:
        return mesh_dir_pkg + "/runtime"
    raise RuntimeError(
        f"cannot uniquely identify skeletal mesh in {phys_dir}; "
        f"declare skeletal_mesh_path explicitly: {candidates or names}"
    )


def _actor_entry(slot, asset_id, by_id, snap):
    rec = by_id[asset_id]
    su = rec["runtime_backends"]["spear_unreal"]
    def phys(package):
        if not package.startswith("/Game/"):
            raise RuntimeError(f"unsupported actor package: {package}")
        p = os.path.join(snap, package.removeprefix("/Game/") + ".uasset")
        if not os.path.isfile(p):
            raise RuntimeError(f"missing physical source: {p}")
        return p

    if rec.get("entity_class") == "rigid_object":
        mesh = su["static_mesh_object_path"]
        package = mesh.split(".", 1)[0]
        return {
            "asset_id": asset_id,
            "entity_class": "rigid_object",
            "entity_instance_id": f"{slot}_actor",
            "profile_alias": asset_id,
            "revision": rec["revision"],
            "source_slot_id": slot,
            "physical_authorized_internal_sources": {"static_mesh": phys(package)},
            "ue_binding": {
                "static_mesh_binding": su["static_mesh_binding"],
                "static_mesh_object_path": mesh,
                "static_mesh_package": package,
            },
        }
    bp = su["blueprint_class_path"]
    bp_pkg = bp.split(".", 1)[0]
    mesh_pkg = _mesh_package_for(rec, snap)
    mesh_name = mesh_pkg.rsplit("/", 1)[-1]

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
                "derivation": ("explicit registry mesh path" if su.get("skeletal_mesh_path")
                               else "legacy unique sibling mesh; actual Blueprint mesh is checked at spawn"),
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
