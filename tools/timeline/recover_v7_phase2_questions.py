#!/usr/bin/env python3
"""Regenerate actual QA and export seals without rerendering completed media."""
import argparse,json
from pathlib import Path
from avengine.timeline.v7_phase2_question_recovery import regenerate_questions


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--source-parent',type=Path,required=True);p.add_argument('--output-native',type=Path,required=True);p.add_argument('--requested-conditions-json',type=Path);a=p.parse_args()
    m=json.loads(a.manifest.read_text());owned=Path(m['output_root']).resolve();source=a.source_parent.resolve();output=a.output_native.resolve()
    if not output.is_relative_to(owned):raise ValueError('recovery output must be in the caller-owned manifest root')
    authorized=[owned,*[Path(x).resolve()for x in m.get('authorized_fresh_source_roots',[])]]
    if not any(source.is_relative_to(x)for x in authorized):raise ValueError('native predecessor outside explicitly authorized roots')
    import os
    from avengine.timeline.v7_demand import SELECTION_FEEDBACK_ENV
    os.environ[SELECTION_FEEDBACK_ENV]=str(owned/'inputs/demand_state.json')
    conditions=json.loads(a.requested_conditions_json.read_text())if a.requested_conditions_json else None
    result=regenerate_questions(source,output,requested_conditions=conditions)
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
