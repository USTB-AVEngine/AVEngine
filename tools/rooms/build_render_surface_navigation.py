"""Build reusable CPU navigation from a room's declared render surface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avengine.rooms.navigation_preparation import prepare_render_surface_navigation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--room-manifest', required=True, type=Path)
    parser.add_argument('--stage-config', type=Path, help='Existing stage orientation for a scan or other native surface')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--runtime-prefix', required=True, type=Path)
    parser.add_argument('--magnum-site', required=True, type=Path)
    parser.add_argument('--rlr-sdk-root', required=True, type=Path)
    args = parser.parse_args()
    report = prepare_render_surface_navigation(
        args.room_manifest, args.output, stage_config=args.stage_config,
        runtime_prefix=args.runtime_prefix, magnum_python_site=args.magnum_site,
        rlr_sdk_root=args.rlr_sdk_root)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
