#!/usr/bin/env python3
"""Write bounded automatic QA reports for an M2 research candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avengine.contracts.json_io import load_json
from avengine.m2.actions import read_baked_actions_npz
from avengine.m2.glb import load_glb
from avengine.m2.habitat import build_habitat_asset_mapping_from_rebase_report
from avengine.m2.qa import audit_m2_candidate


SEMANTIC_JOINT_MAP = {
    "body": "beagle Pelvis",
    "head": "beagle Head",
    "muzzle": "beagle Xtra Mouth",
    "paw_front_left": "beagle L Finger0",
    "paw_front_right": "beagle R Finger0",
    "paw_hind_left": "beagle L Toe0",
    "paw_hind_right": "beagle R Toe0",
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--actions-npz", type=Path, required=True)
    parser.add_argument("--rebase-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    document = load_glb(args.visual_glb)
    actions = read_baked_actions_npz(args.actions_npz)
    mapping = build_habitat_asset_mapping_from_rebase_report(
        document, load_json(args.rebase_report)
    )
    result = audit_m2_candidate(
        document,
        actions,
        mapping,
        semantic_joint_map=SEMANTIC_JOINT_MAP,
    )
    paths = {
        "static_geometry": output / "static_geometry.json",
        "deformation": output / "deformation.json",
        "animation": output / "animation.json",
    }
    _write_json(paths["static_geometry"], result.static_geometry)
    _write_json(paths["deformation"], result.deformation)
    _write_json(paths["animation"], result.animation)
    print(
        json.dumps(
            {
                "automatic_qa_status": "pass",
                "qualification_state": "research_candidate",
                "qualification_claim": False,
                "human_visual_review_required": True,
                "known_limitations": result.animation["known_limitations"],
                "reports": {key: str(path) for key, path in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
