#!/usr/bin/env python3
"""Plan a CPU-only QA-05 reference group.

The tool validates a four-object fixed-camera plan, a private pixel-centroid
selector proof (leftmost or named-reference), and two timing-only audio columns.
It never starts UE or RLR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.binding_group_reference import (  # noqa: E402
    REFERENCE_QUERY_KINDS,
    ReferenceNativeError,
    plan_reference_group,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--second-plan", type=Path)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--v0-capture-root", type=Path)
    parser.add_argument("--v1-capture-root", type=Path)
    parser.add_argument(
        "--pixel-centers",
        type=Path,
        help="CPU fixture JSON with {v0:{source1..source4},v1:{source1..source4}}",
    )
    parser.add_argument("--group-id", default="visual_conditioned_reference_group_v1")
    parser.add_argument("--world-id", default="world_visual_conditioned_reference_0001")
    parser.add_argument(
        "--selector-kind",
        choices=REFERENCE_QUERY_KINDS,
        help="optional assertion that the request uses the selected visual selector",
    )
    args = parser.parse_args(argv)
    try:
        centers = None
        if args.pixel_centers is not None:
            centers = json.loads(args.pixel_centers.read_text(encoding="utf-8"))
        summary = plan_reference_group(
            base_plan_path=args.base_plan,
            second_plan_path=args.second_plan,
            request_path=args.request,
            output_root=args.output_root,
            v0_capture_root=args.v0_capture_root,
            v1_capture_root=args.v1_capture_root,
            pixel_centers_by_variant=centers,
            group_id=args.group_id,
            world_id=args.world_id,
        )
        actual_selector = summary["query"]["visual_selector"]["kind"]
        if args.selector_kind is not None and actual_selector != args.selector_kind:
            raise ReferenceNativeError(
                f"request visual selector is {actual_selector}, expected {args.selector_kind}"
            )
    except (ReferenceNativeError, OSError, RuntimeError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "native_execution": summary["native_execution"],
                "rlr_execution": summary["rlr_execution"],
                "visual_plan_relation": summary["visual_plan_relation"],
                "visual_selector": summary["query"]["visual_selector"],
                "audio_columns": summary["audio_columns"],
                "group_spec": summary["group_spec"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
