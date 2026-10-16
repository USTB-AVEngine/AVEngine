#!/usr/bin/env python3
"""Audit a no-render post-dedup modality re-selection for this acceptance batch."""
import argparse,json
from pathlib import Path
from avengine.timeline.v7_admitted_pool_reselection import balanced_batch
def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    m=json.loads((a.output_root/'inputs/ACCEPTANCE_PLAN.json').read_text())
    assert m['dataset_role']=='acceptance_only'and m['training_dataset_merge']is False and m['CCI_transfer']is False
    receipts=[]
    for task in m['parents']:
     f=a.output_root/'evidence/jobs'/f"{task['slot']:03d}.json"
     if not f.exists():continue
     r=json.loads(f.read_text())
     if r.get('status')!='passed':continue
     pointer=json.loads(Path(r['export']['source_seal']).read_text());recipe=pointer['recipe']['input_recipe'];row=recipe.get('row',recipe.get('seed_row'))
     if any(v.get('action','').startswith('removed_')for v in row.get('element_degradations',[])):continue
     if any(row.get('source_classes',{}).get(aid)!=['footsteps']for aid in row.get('phase2_include_footsteps_asset_ids',[])):continue
     if row.get('acceptance_allow_partial_families')is True:continue
     receipts.append(r)
    rows,report=balanced_batch(receipts,a.output_root,read_only_source_roots=m.get('peer_output_roots',[]),workers=a.workers,minimum_houses_per_family=int(m.get('minimum_family_distinct_houses',0)),family_minimums={f:m['motion_quartet_final_targets']['minimum_independent_questions_per_family']for f in m.get('motion_quartet_final_targets',{}).get('qa_ids',[])})
    print(json.dumps({k:v for k,v in report.items()if k!='parents'},ensure_ascii=False))

if __name__=='__main__':main()
