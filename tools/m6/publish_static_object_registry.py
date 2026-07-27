#!/usr/bin/env python3
"""Publish or verify one fail-closed M6 static-object research registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from avengine.m6.static_objects import (
    StaticObjectRegistrationError,
    publish_static_object_entity_registry,
    verify_static_object_entity_registry,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser(
        "publish", help="append a rigid-object row and publish without replacement"
    )
    publish.add_argument("--base-registry", type=Path, required=True)
    publish.add_argument("--admission-batch", type=Path, required=True)
    publish.add_argument("--instance-id", required=True)
    publish.add_argument("--marker-visual-approval", type=Path, required=True)
    publish.add_argument("--registry-revision", required=True)
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument(
        "--workspace-root",
        action="append",
        type=Path,
        required=True,
        help=(
            "Trusted input/output root; repeat for AVEngine, resolved SPEAR, "
            "recorded Blender, ISNet model, and configured/resolved Python "
            "binary roots."
        ),
    )

    verify = subparsers.add_parser(
        "verify", help="rehash the registry and its complete static evidence closure"
    )
    verify.add_argument("registry", type=Path)
    verify.add_argument("--entity-asset-id", required=True)
    verify.add_argument("--entity-revision", required=True)
    verify.add_argument(
        "--workspace-root",
        action="append",
        type=Path,
        required=True,
        help=(
            "Trusted evidence root; repeat to include recorded Blender, "
            "SPEAR, ISNet model, and Python binary roots."
        ),
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.command == "verify":
            entity = verify_static_object_entity_registry(
                registry_path=arguments.registry,
                entity_asset_id=arguments.entity_asset_id,
                entity_revision=arguments.entity_revision,
                workspace_roots=arguments.workspace_root,
            )
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "registry": str(arguments.registry.resolve()),
                        "entity_asset_id": entity["entity_asset_id"],
                        "entity_revision": entity["revision"],
                        "admission_state": entity["admission_state"],
                        "formal_dataset_registration_authorized": entity[
                            "admission_evidence"
                        ]["formal_dataset_registration_authorized"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        output = publish_static_object_entity_registry(
            base_registry_path=arguments.base_registry,
            admission_batch_path=arguments.admission_batch,
            instance_id=arguments.instance_id,
            marker_visual_approval_path=arguments.marker_visual_approval,
            output_path=arguments.output,
            registry_revision=arguments.registry_revision,
            workspace_roots=arguments.workspace_root,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        static = next(
            item
            for item in payload["entities"]
            if item.get("entity_asset_id") == arguments.instance_id
        )
    except (
        FileExistsError,
        OSError,
        StaticObjectRegistrationError,
        StopIteration,
    ) as error:
        print(f"M6_STATIC_OBJECT_REGISTRATION_FAILED {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "registry": str(output),
                "entity_asset_id": static["entity_asset_id"],
                "entity_revision": static["revision"],
                "admission_state": static["admission_state"],
                "formal_dataset_registration_authorized": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
