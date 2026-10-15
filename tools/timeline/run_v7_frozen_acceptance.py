#!/usr/bin/env python3
"""Freeze committed code inside the authorised worktree before launching a parent."""
import argparse,fcntl,json,os,pathlib,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--worker',required=True,type=int);a=p.parse_args()
m=json.loads(pathlib.Path(a.manifest).read_text());work=pathlib.Path(m['worktree']).resolve();commit=m['code_commit'];assert len(commit)==40 and all(c in '0123456789abcdef'for c in commit)
base=work/'src/.v7_snapshots';base.mkdir(exist_ok=True);snapshot=base/commit
with(base/(commit+'.lock')).open('a+')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if not(snapshot/'FROZEN_COMMIT').is_file():
  snapshot.mkdir(exist_ok=True)
  archive=subprocess.Popen(['git','archive','--format=tar',commit],cwd=work,stdout=subprocess.PIPE)
  subprocess.run(['tar','-x','-C',str(snapshot)],stdin=archive.stdout,check=True);archive.stdout.close()
  if archive.wait():raise RuntimeError('committed source archive failed')
  (snapshot/'FROZEN_COMMIT').write_text(commit+'\n')
 assert(snapshot/'FROZEN_COMMIT').read_text().strip()==commit
runtime=os.environ.copy();runtime['PYTHONPATH']=str(snapshot/'src')+':'+str(snapshot)+':'+runtime.get('PYTHONPATH','');runtime['AVENGINE_FROZEN_SOURCE_ROOT']=str(snapshot);runtime['AVENGINE_FROZEN_COMMIT']=commit
probe=subprocess.check_output([sys.executable,'-c','import avengine;print(avengine.__file__)'],env=runtime,text=True).strip();assert pathlib.Path(probe).is_relative_to(work/'src')
evidence=pathlib.Path(m['output_root'])/'evidence'/f'FROZEN_IMPORT_{a.worker:03d}.json';evidence.write_text(json.dumps({'commit':commit,'snapshot':str(snapshot),'avengine_file':probe,'source_worktree':str(work)},indent=2)+'\n')
os.execve(sys.executable,[sys.executable,'-u',str(snapshot/'tools/timeline/run_v7_acceptance.py'),'--manifest',a.manifest,'--worker',str(a.worker)],runtime)
