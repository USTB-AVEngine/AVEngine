#!/usr/bin/env python3
"""Ship actual commit/tree/blob objects for Git-dependent archived generators.

No new commit is invented in CCI. The isolated repository is shallow at the
actual 48g commit, and read-tree only installs its index, preserving code files.
"""
import argparse,hashlib,json,os,subprocess
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def command(root,*args,**kwargs):
 env=dict(os.environ);config=root/'.git/CCI_SAFE_CONFIG'
 if config.is_file():env['GIT_CONFIG_GLOBAL']=str(config)
 return subprocess.run(['git','-C',str(root),*args],check=True,env=env,**kwargs)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['pack','install']);p.add_argument('--code-root',required=True);p.add_argument('--pack',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();root=Path(a.code_root).resolve();pack=Path(a.pack)
 if a.mode=='pack':
  commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD']).decode().strip();objects={item.split(b'\t',1)[0].split()[2].decode() for item in subprocess.check_output(['git','-C',str(root),'ls-tree','-r','-t','-z',commit]).split(b'\0') if item};objects.add(commit);objects.add(subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD^{tree}']).decode().strip())
  with pack.open('wb') as f:command(root,'pack-objects','--quiet','--stdout',input=('\n'.join(sorted(objects))+'\n').encode(),stdout=f)
  Path(a.manifest).write_text(json.dumps({'source_commit':commit,'source_tree':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD^{tree}']).decode().strip(),'object_count':len(objects),'pack_sha256':sha(pack),'pack_bytes':pack.stat().st_size,'scope':'actual objects for this commit and tree only; shallow history'},indent=2)+'\n')
 else:
  manifest=json.loads(Path(a.manifest).read_text());commit=manifest['source_commit']
  if sha(pack)!=manifest['pack_sha256']:raise ValueError('Git object pack SHA mismatch')
  marker=root/'.git/CCI_SNAPSHOT.json'
  if (root/'.git').exists() and not marker.exists():raise ValueError('refuse an existing repository not owned by this release installer')
  if not (root/'.git').exists():command(root,'init','--quiet')
  marker.write_text(json.dumps({**manifest,'installer_state':'installing'},indent=2)+'\n')
  # AFS maps ownership to the user's numeric UID while this container is root.
  # Older Git accepts safe.directory only through global-file scope; redirect
  # that scope for our child commands to this private repository file.
  (root/'.git/CCI_SAFE_CONFIG').write_text('[safe]\n\tdirectory = '+json.dumps(str(root))+'\n')
  with pack.open('rb') as f:command(root,'index-pack','--stdin',stdin=f,stdout=subprocess.DEVNULL)
  command(root,'update-ref','--no-deref','HEAD',commit)
  (root/'.git/shallow').write_text(commit+'\n');command(root,'read-tree',commit)
  marker.write_text(json.dumps(manifest,indent=2)+'\n')
  actual=command(root,'rev-parse','HEAD',stdout=subprocess.PIPE).stdout.decode().strip()
  if actual!=commit:raise RuntimeError('installed Git commit does not match actual 48g source')
  print(json.dumps({'source_commit':commit,'git_status':command(root,'status','--porcelain',stdout=subprocess.PIPE).stdout.decode(),'pack_sha256':manifest['pack_sha256']}))
if __name__=='__main__':main()
