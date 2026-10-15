"""Serialize task-manifest appends from room and required-class coordinators."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import fcntl,json,os

@contextmanager
def mutation_lock(path):
    with Path(str(path)+'.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);yield

def atomic_write(path,data):
    path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

def append_task(path,task):
    with mutation_lock(path):
        data=json.loads(Path(path).read_text());task=deepcopy(task);task['slot']=len(data['parents']);data['parents'].append(task);atomic_write(path,data);return task
