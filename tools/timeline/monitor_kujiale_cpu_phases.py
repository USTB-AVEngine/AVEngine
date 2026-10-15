#!/usr/bin/env python3
"""Balance exact own CPU-only workers while native UE jobs keep their cores."""
import argparse,json,time
from pathlib import Path
from avengine.timeline.v7_kujiale_cpu_balance import rebalance
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--stop-file',type=Path,required=True);a=p.parse_args()
    while not a.stop_file.exists():
        result=rebalance(a.output_root)
        if result['changes']:
            with(a.output_root/'evidence/cpu_balance_actions.jsonl').open('a')as f:f.write(json.dumps(result)+'\n')
        time.sleep(10)
