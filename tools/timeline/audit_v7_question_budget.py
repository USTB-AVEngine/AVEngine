#!/usr/bin/env python3
"""Audit actual v7 logical items, modality ratios, family floors and CF proofs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from avengine.timeline.v7_production_contract import audit_batch_budget,family_minima_for_policy
from avengine.timeline.continuous_pilot import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--window-list',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scale',choices=('v7_S','full'),required=True);p.add_argument('--split',choices=('train','dev','test'),required=True)
    p.add_argument('--counterfactual-proof',type=Path,action='append',default=[]);a=p.parse_args()
    policy=json.loads(a.policy.read_text())
    windows=[]
    for value in json.loads(a.window_list.read_text()):
        path=Path(value);public=path/'questions.json'
        if not public.exists():public=path/'public_questions.json'
        windows.append((json.loads(public.read_text()),json.loads((path/'private_answers.json').read_text())))
    binding=[]
    for path in a.counterfactual_proof:
        proof=json.loads(path.read_text())
        if proof['status']!='pass' or not all(c['pass'] for c in proof['checks']):raise ValueError('counterfactual proof failed')
        binding.extend(proof['verified_binding_logical_question_ids'])
    published={q['qa_id'] for public,_ in windows for q in public}
    minima=family_minima_for_policy(policy,a.scale,a.split,published)
    report=audit_batch_budget(windows,minima,binding)
    write_json(a.output,report);print(json.dumps(report,ensure_ascii=False))
    if report['status']!='pass':raise SystemExit(2)

if __name__=='__main__':main()
