#!/usr/bin/env python3
"""Bind accepted generator/assets/CCI runtime to a new configuration; never launch jobs."""
import argparse
import json
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from avengine.qa.frozen_panel_versions import read, validate_versions, digest, versions_identity, require_canary_acceptance


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--versions', type=Path, required=True)
    p.add_argument('--prepared-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--output-parent', type=Path, required=True)
    p.add_argument('--canary-comparison', type=Path)
    p.add_argument('--canary-verification', type=Path)
    p.add_argument('--canary-sizes', type=Path)
    a = p.parse_args()
    c = read(a.config)
    v = read(a.versions)
    c.update(production_versions=v, registry=v['assets']['registry'],
             template_plan=v['assets']['template_plan'], runtime=v['render_environment']['runtime'],
             prepared_geometry_root=str(a.prepared_root.resolve()), retain_compact=True,
             allowed_output_root=str(a.output_parent.resolve()))
    validate_versions(c, require_host=False)
    if c.get('split') == 'test':
        if not all([a.canary_comparison, a.canary_verification, a.canary_sizes]):
            raise ValueError('formal binding needs --canary-comparison/--canary-verification/--canary-sizes')
        c['canary_acceptance'] = {'production_versions_sha256': versions_identity(v),
            **{name: {'path': str(path.resolve()), 'sha256': digest(path)} for name, path in
               [('comparison', a.canary_comparison), ('verification', a.canary_verification), ('sizes', a.canary_sizes)]}}
        require_canary_acceptance(c)
    parents = []
    for house in c['houses']:
        x = read(a.prepared_root / house['house'] / 'prepared.json')
        if x['status'] != 'pass' or len(x['groups']) != c['groups_per_house']:
            raise ValueError('incomplete prepared house')
        parents += [g['physical_parent_sha256'] for g in x['groups']]
    if len(parents) != len(set(parents)):
        raise ValueError('prepared independent parent duplication')
    c['reserved_preflight_physical_parent_sha256'] = parents
    c['prepared_geometry_sha256'] = {h['house']: digest(a.prepared_root / h['house'] / 'prepared.json') for h in c['houses']}
    with a.output.open('x') as f:
        json.dump(c, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': 'pass', 'config': str(a.output), 'parents': len(parents), 'jobs_launched': False}))


if __name__ == '__main__':
    main()
