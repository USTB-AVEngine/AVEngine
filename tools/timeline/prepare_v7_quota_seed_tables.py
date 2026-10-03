#!/usr/bin/env python3
"""Version v7-S/full requests with balanced speaker, spatial and class quotas."""
from __future__ import annotations
import argparse,json,os,uuid
from pathlib import Path
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_production_contract import validate_seed_tables
from avengine.timeline.v7_scene_quotas import add_quotas,quota_summary,QUOTA_INPUT_VERSION
NAMES=('v7_S_train.jsonl','v7_S_dev.jsonl','v7_full_train_seed_plan.jsonl','v7_full_dev_seed_plan.jsonl','v7_full_test_seed_plan.jsonl')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    tables={name:add_quotas([json.loads(line) for line in (a.previous/name).read_text().splitlines() if line.strip()]) for name in NAMES}
    receipt=validate_seed_tables(tables[NAMES[0]],tables[NAMES[1]],tables[NAMES[2]])
    if tables[NAMES[3]][:100]!=tables[NAMES[1]]:raise ValueError('small/full dev prefix differs')
    a.output.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        payload=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n' for row in rows)
        path=a.output/name
        if path.exists():
            if path.read_text()!=payload:raise ValueError('existing quota seeds differ')
        else:
            pending=path.with_name('.'+name+'.'+uuid.uuid4().hex+'.partial')
            with pending.open('x') as f:f.write(payload);f.flush();os.fsync(f.fileno())
            os.link(pending,path)
    validation={**receipt,'schema':QUOTA_INPUT_VERSION,'dev_prefix_equal':True,'source':str(a.previous.resolve()),
        'formal_generation_ready':False,'blocking_reasons':['moving_RIR_energy_admission_failed','native_quota_layout_and_dry_family_admission_pending'],
        'quota_counts':{name:quota_summary(rows) for name,rows in tables.items()}}
    if (a.output/'validation.json').exists():
        if json.loads((a.output/'validation.json').read_text())!=validation:raise ValueError('validation differs')
    else:write_json(a.output/'validation.json',validation)
    print(json.dumps({k:v for k,v in validation.items() if k!='quota_counts'},ensure_ascii=False))
if __name__=='__main__':main()
