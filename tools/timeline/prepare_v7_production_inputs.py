#!/usr/bin/env python3
"""Freeze v7 room, sound, dialogue and motion input authority without rendering.

Usage: python tools/timeline/prepare_v7_production_inputs.py --config CONFIG --output OUTPUT
Both paths are explicit. The source manifests, registrations and raw assets are
read-only; no CCI synchronization or training export is performed.
"""
import argparse
import json
from pathlib import Path
from avengine.timeline.v7_production_inputs import prepare_contract, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    path = Path(a.output)
    if path.exists():
        raise SystemExit('output already exists; use a new immutable snapshot path')
    result = prepare_contract(read_json(a.config))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'rooms': len(result['rooms']), 'audio': result['audio'], 'dialogue': result['dialogue']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
