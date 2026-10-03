#!/usr/bin/env python3
"""Write v7-S and full-set seed tables without changing existing inputs."""
from __future__ import annotations
import argparse
import json
import os
import uuid
from pathlib import Path
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_production_contract import seed_rows,validate_seed_tables,VERSION

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--houses',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-train-count',type=int,default=7000)
    a=p.parse_args(); houses=json.loads(a.houses.read_text())['houses']
    train=seed_rows('train',400,houses);dev=seed_rows('dev',100,houses)
    full=seed_rows('train',a.full_train_count,houses)
    full_dev=seed_rows('dev',1500,houses);full_test=seed_rows('test',1500,houses)
    receipt=validate_seed_tables(train,dev,full)
    a.output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('v7_S_train.jsonl',train),('v7_S_dev.jsonl',dev),('v7_full_train_seed_plan.jsonl',full),('v7_full_dev_seed_plan.jsonl',full_dev),('v7_full_test_seed_plan.jsonl',full_test)]:
        path=a.output/name
        payload=''.join(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n' for x in rows)
        # No overwrite, including when a caller changes admitted-house scope.
        if path.exists():
            if path.read_text()!=payload:raise ValueError('existing seed table differs')
        else:
            pending=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.partial')
            with pending.open('x') as f:
                f.write(payload);f.flush();os.fsync(f.fileno())
            os.link(pending,path)
    validation={**receipt,'schema':VERSION,'input_houses':str(a.houses.resolve())}
    if (a.output/'validation.json').exists():
        if json.loads((a.output/'validation.json').read_text())!=validation:raise ValueError('existing seed validation changed')
    else:write_json(a.output/'validation.json',validation)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
