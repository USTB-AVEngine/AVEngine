"""Atomically replace explicit live counters; never replace sealed artifacts."""
import json,os,uuid
from pathlib import Path


def write_live_status(path,value):
    path=Path(path)
    if 'ue_round8' not in path.parts or path.name not in {'live_result.json'} or path.is_symlink():raise ValueError('only task-owned live counters may be replaced')
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x') as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
