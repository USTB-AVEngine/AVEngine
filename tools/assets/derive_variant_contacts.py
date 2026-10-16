#!/usr/bin/env python3
"""Derive generic package anchors and actor-space four-paw contacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.assets.variant_contacts import (
    VariantContactError,
    derive_variant_contact_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--actions-npz", type=Path, required=True)
    parser.add_argument("--rebase-report", type=Path, required=True)
    parser.add_argument("--emitter-anchors-output", type=Path, required=True)
    parser.add_argument("--contact-phases-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = derive_variant_contact_artifacts(
            spec_path=args.spec,
            visual_glb=args.visual_glb,
            actions_npz=args.actions_npz,
            rebase_report=args.rebase_report,
            emitter_anchors_output=args.emitter_anchors_output,
            contact_phases_output=args.contact_phases_output,
        )
    except VariantContactError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
