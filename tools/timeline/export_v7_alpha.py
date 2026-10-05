#!/usr/bin/env python3
"""Package immutable admitted alpha parents and training-ready relative media."""
from pathlib import Path
import argparse,json
from avengine.timeline.v7_alpha_delivery import export_alpha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seeds',type=Path,required=True)
    p.add_argument('--seal-roots',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-parents',type=int,required=True);p.add_argument('--seal-parent-map',type=Path);a=p.parse_args()
    result=export_alpha(a.seeds,a.seal_roots,a.output,a.expected_parents,a.seal_parent_map)
    print(json.dumps({k:result[k] for k in ['status','independent_parent_scenes','formal_windows','formal_logical_questions','modality_percent','binding_percent']},ensure_ascii=False))
if __name__=='__main__':main()
