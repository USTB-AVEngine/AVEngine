#!/usr/bin/env python3
"""Resume only selection of an immutable completed new acceptance parent."""
import argparse,json,os,sys
from pathlib import Path
W=Path(__file__).resolve().parents[2];sys.path[:0]=[str(W/'src'),str(W)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source-attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
    from tools.timeline.generate_v7_demand_scenes import load_global
    from avengine.timeline.v7_selection_resume import resume_completed_selection
    result=resume_completed_selection(a.source_attempt,a.output,load_global(a.config));print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status']=='passed'else 3 if result['status'].startswith('paused')else 2
if __name__=='__main__':raise SystemExit(main())
