#!/usr/bin/env python3
"""Emit Room C static furniture and seat semantics after the shared builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = args.output_root.expanduser().resolve()
    if not output.is_dir():
        raise SystemExit(f"builder output root is missing: {output}")
    target = output / "object_semantics.json"
    if target.exists():
        raise SystemExit(f"refusing to replace semantics output: {target}")
    objects = []
    for item in spec.get("semantic_objects", []):
        objects.append(dict(item, source="room_spec"))
    for anchor_id, position in spec.get("anchors", {}).items():
        objects.append({
            "object_id": anchor_id,
            "category": "anchor",
            "zone_id": None,
            "position_m": list(position),
            "navigation_role": "semantic_point",
            "static": True,
            "seat_points": [],
            "source": "room_spec"
        })
    manifest = {
        "kind": "avengine_room_c_object_semantics",
        "status": "research_candidate",
        "qualification_claim": False,
        "room_spec_id": spec["room_spec_id"],
        "coordinate_system": "room_local_xy_plus_z",
        "objects": objects,
        "interaction_boundary": dict(spec.get("semantic_boundary", {})),
    }
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    handoff = {
        "kind": "avengine_room_c_handoff",
        "status": "research_candidate",
        "qualification_claim": False,
        "room_id": spec["room_spec_id"],
        "room_family_id": spec.get("room_family_id"),
        "topology_family": spec.get("topology_family"),
        "artifacts": {
            "blend": str(output / f"{spec['room_spec_id']}.blend"),
            "visual_glb": str(output / "visual" / f"{spec['room_spec_id']}.glb"),
            "collision_glb": str(output / "visual" / f"{spec['room_spec_id']}_collision.glb"),
            "usd": str(output / "usd" / f"{spec['room_spec_id']}.usda"),
            "object_semantics": str(target),
            "functional_anchors": str(output / "functional_anchors.json"),
        },
        "native_execution": "pending_root_spear_ue",
        "interaction_boundary": dict(spec.get("semantic_boundary", {})),
    }
    (output / "room_handoff.json").write_text(
        json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "research_candidate", "object_count": len(objects), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
