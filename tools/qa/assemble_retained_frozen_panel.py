#!/usr/bin/env python3
"""Delegate scoring assembly to the unchanged scorer, then assemble minimal witnesses."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from avengine.qa.frozen_panel_retention import read,sha,write
from compact_frozen_panel import freeze


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shard-root',type=Path,action='append',required=True)
    p.add_argument('--expected-groups',type=int,default=720);a=p.parse_args()
    root=a.root.resolve();shards=[s.resolve() for s in a.shard_root]
    if root.exists():raise ValueError('assembly output must be new')
    groups=[];sources=[];names=set()
    for shard in shards:
        index=read(shard/'retention/manifest.json')
        for line in (shard/'retention/COMMITMENTS.sha256').read_text().splitlines():
            digest,relative=line.split('  ',1)
            if sha(shard/'retention'/relative)!=digest:raise ValueError('shard lossless witness commitment changed')
        for relative in index['groups']:
            if relative in names:raise ValueError('retained group overlaps shards')
            names.add(relative);groups.append(relative)
        sources.append({'root':str(shard),'retention_commitment_sha256':sha(shard/'retention/COMMITMENTS.sha256')})
    if len(groups)!=a.expected_groups:raise ValueError('retained group total differs')
    cmd=[sys.executable,str(REPO/'tools/qa/score_frozen_panel.py'),'assemble','--root',str(root),'--expected-groups',str(a.expected_groups)]
    for shard in shards:cmd+=['--shard-root',str(shard)]
    subprocess.run(cmd,check=True)  # Only the original scorer reads answers.
    retention=root/'retention';retention.mkdir(mode=0o700)
    for shard in shards:
        for directory in ['blobs','groups','prune_receipts']:
            source=shard/'retention'/directory
            if not source.exists():continue
            (retention/directory).mkdir(exist_ok=True)
            for file in source.iterdir():
                if file.is_symlink() or not file.is_file():raise ValueError('unexpected retention directory entry')
                target=retention/directory/file.name
                if target.exists():
                    if sha(target)!=sha(file):raise ValueError('retained content-address collision')
                else:
                    try:os.link(file,target)
                    except OSError:shutil.copyfile(file,target)
    write(retention/'manifest.json',{'schema':'frozen_panel_lossless_retention_v1','groups':sorted(groups),
        'source_root':str(root),'original_score_and_truth_unchanged':True,'shard_witness_sources':sources})
    freeze(root)
    print(json.dumps({'status':'pass','groups':len(groups),'root':str(root),'scorer_unchanged':True}))
if __name__=='__main__':main()
