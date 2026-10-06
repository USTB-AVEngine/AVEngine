"""Read-only v7 P19 status: actual claimant, all rank events and fresh CPU progress."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from avengine.timeline.v7_p19_progress import inspect_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--active-root', type=Path, help='optional isolated fixture lease directory')
    args = parser.parse_args()
    print(json.dumps(inspect_plan(args.plan, active_root=args.active_root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
