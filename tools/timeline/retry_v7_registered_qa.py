#!/usr/bin/env python3
"""Start a fresh QA qualification while preserving original native checkpoints."""
from __future__ import annotations
import argparse,copy,datetime,json,os,shutil
from pathlib import Path
from avengine.timeline.v7_kujiale_manifest import append_task

def load(path):return json.loads(Path(path).read_text())
def clone_native(source,target):
    source=Path(source).resolve();target=Path(target).resolve()
    if target.exists():raise ValueError('new QA retry root must not already exist')
    target.mkdir(parents=True);shutil.copy2(source/'input_recipe.json',target/'input_recipe.json');stages=['dry','visual_v0','visual_v1',*[f'audio_{v}_{a}'for v in ['v0','v1']for a in ['a0','a1']]]
    for stage in stages:
        marker=source/(stage+'_completed.json');info=load(marker);attempt=source/info['attempt']
        if any(not(attempt/name).is_file()for name in info['required_files']):raise ValueError('source native stage is incomplete')
        shutil.copytree(attempt,target/info['attempt'],copy_function=os.link,ignore=shutil.ignore_patterns('registry_namespace*.json'));shutil.copy2(marker,target/marker.name)
    cache=source/'rir_cache'
    if cache.exists():
        (target/'rir_cache').mkdir()
        for path in cache.iterdir():
            if path.is_file():
                if path.suffix=='.npy':os.link(path,target/'rir_cache'/path.name)
                else:shutil.copy2(path,target/'rir_cache'/path.name)
    (target/'native_reuse_provenance.json').write_text(json.dumps({'source':str(source),'target':str(target),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reused':'complete original native/dry/audio stages only; original QA pools/checkpoints unchanged','original_qa_reused':False},indent=2)+'\n')

def append_retry(manifest_path,slot,native_root):
    manifest=load(manifest_path);root=Path(manifest['output_root']).resolve();task=copy.deepcopy(manifest['parents'][slot]);source=Path(task.get('native_output_root')or root/'native')/'parents'/task['row']['scene_id'];native_root=Path(native_root).resolve()
    if not native_root.is_relative_to(root):raise ValueError('QA retry escaped task root')
    target=native_root/'parents'/task['row']['scene_id'];clone_native(source,target);task['slot']=len(manifest['parents']);task['native_output_root']=str(native_root);task['row']=load(source/'input_recipe.json')['seed_row'];task=append_task(manifest_path,task);planned=root/'inputs/planned'/f'{task["slot"]:03d}.json';planned.write_text(json.dumps(task['row'],ensure_ascii=False,indent=2)+'\n');return task['slot']

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--slot',type=int,required=True);p.add_argument('--native-root',type=Path,required=True);a=p.parse_args();print(append_retry(a.manifest,a.slot,a.native_root))
if __name__=='__main__':main()
