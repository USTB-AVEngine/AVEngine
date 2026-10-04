#!/usr/bin/env python3
"""Project fixed S parent seeds onto measured alpha layouts, preserving prefixes.

This creates inputs only. Native render/QA qualification must still succeed for
all parents before an alpha release can be exported.
"""
from pathlib import Path
import argparse,json
from collections import Counter
from avengine.timeline.v7_alpha_sampling import alpha_row
from avengine.timeline.continuous_pilot import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['train-seeds','dev-seeds','layouts','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();layouts=json.loads(a.layouts.read_text())['houses'];by_house={h['house']:h for h in layouts if h['status']=='pass'}
    inputs={split:[json.loads(l) for l in path.read_text().splitlines() if l.strip()] for split,path in [('train',a.train_seeds),('dev',a.dev_seeds)]}
    if len(inputs['train'])!=400 or len(inputs['dev'])!=100:raise ValueError('fixed S must contain 400 train and 100 dev parents')
    missing=sorted({r['house'] for rows in inputs.values() for r in rows}-set(by_house))
    if missing:raise ValueError('native static layouts missing: '+', '.join(missing))
    if a.output.exists():raise ValueError('select a fresh seed output directory')
    a.output.mkdir(parents=True);summary={'schema':'v7-S-alpha-seed-projection-v1','inputs_only_not_formal_data':True,'splits':{}}
    for split,originals in inputs.items():
        rows=[alpha_row(row,by_house[row['house']],i,force_quartet=False) for i,row in enumerate(originals)]
        for original,row in zip(originals,rows,strict=True):
            for field in ['scene_id','scene_seed','seeds','house','split','window_start_ticks','members','question_quota_base']:
                if original[field]!=row[field]:raise ValueError('original parent prefix changed: '+field)
        text=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)
        (a.output/f'v7_S_alpha_{split}.jsonl').write_text(text)
        # The full alpha table must start with these exact rows. Unassigned
        # remainder of the existing full reserve is not released as runnable.
        (a.output/f'v7_full_alpha_{split}_prefix{len(rows)}.jsonl').write_text(text)
        windows=sum(len(r['members'])*r['window_count'] for r in rows)
        summary['splits'][split]={'parents':len(rows),'houses':dict(Counter(r['house'] for r in rows)),
            'member_counts':dict(Counter(len(r['members']) for r in rows)),
            'native_visual_worlds':sum(len({m.split('_')[0] for m in r['members']}) for r in rows),
            'windows':windows,'planned_questions':windows*14,'full_prefix_byte_identical':True}
    write_json(a.output/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
