#!/usr/bin/env python3
"""Extend reproducible v7-S/full seeds with simulation-only diagnostic axes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_production_contract import validate_seed_tables
from avengine.timeline.v7_scene_quotas import add_diagnostic_quotas,diagnostic_summary,DIAGNOSTIC_INPUT_VERSION
from tools.timeline.prepare_v7_quota_seed_tables import NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    tables={n:add_diagnostic_quotas([json.loads(s) for s in (args.previous/n).read_text().splitlines() if s.strip()]) for n in NAMES}
    check=validate_seed_tables(tables[NAMES[0]],tables[NAMES[1]],tables[NAMES[2]])
    if tables[NAMES[3]][:100]!=tables[NAMES[1]]:raise ValueError('small/full dev prefix changed')
    args.output.mkdir(parents=True,exist_ok=False)
    for name,rows in tables.items():
        with (args.output/name).open('x') as stream:
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    write_json(args.output/'validation.json',{**check,'schema':DIAGNOSTIC_INPUT_VERSION,'dev_prefix_equal':True,
        'source':str(args.previous.resolve()),'base_seeds_changed':False,'formal_generation_ready':False,
        'quota_counts':{name:diagnostic_summary(rows) for name,rows in tables.items()}})
    print(json.dumps(check,ensure_ascii=False))


if __name__=='__main__':main()
