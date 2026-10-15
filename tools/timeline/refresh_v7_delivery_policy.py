#!/usr/bin/env python3
"""Recheck latest delivery conditions without changing source seals or media."""
import argparse,datetime,json,time
from pathlib import Path
from avengine.timeline.v7_acceptance_publication import publish_current_parents


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--until-utc');p.add_argument('--interval-s',type=float,default=15.);a=p.parse_args()
    end=datetime.datetime.fromisoformat(a.until_utc)if a.until_utc else None
    last=None
    while True:
        m=json.loads(a.manifest.read_text());result=publish_current_parents(m['output_root']);current=(result['independent_questions'],result['houses'])
        if current!=last:print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'CURRENT_POLICY_EXPORT',current,flush=True);last=current
        if end is None or datetime.datetime.now(datetime.timezone.utc)>=end:break
        time.sleep(a.interval_s)
if __name__=='__main__':main()
