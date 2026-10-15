"""Explicit owner invalidation markers isolate fresh generations."""
from pathlib import Path
import re


def invalidation_marker(path):
    path=Path(path).resolve()
    for parent in (path,*path.parents):
        marker=next(iter(sorted(parent.glob('INVALID_*.md'))),None)
        if marker:return marker
    return None


def assert_new_generation_output(path):
    marker=invalidation_marker(path)
    if marker:raise ValueError('owner-invalidated generation output: '+str(marker)+'; use a new generation directory')


def generation_scene_id(authorized_root,run_root,index):
    safe=lambda value:re.sub('[^A-Za-z0-9_]','_',Path(value).name)
    return 'auto_'+safe(authorized_root)+'_'+safe(run_root)+'_'+str(index).zfill(4)
