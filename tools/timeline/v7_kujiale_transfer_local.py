"""Pull only sealed own releases into own CCI namespace, verify all SHA."""
from pathlib import Path
import subprocess,json,time,shlex,sys,hashlib,os
LOCAL=Path('/home/neos/jzy/codex_runs/20261003-kujiale')
REMOTE='/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/ue_round14'
DEST='/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_s_kujiale_20261005'
def sshcci(script):
 return subprocess.run(['ssh','-o','ControlMaster=no','-o','ControlPath=none','-o','BatchMode=yes','-o','ConnectTimeout=15','cci-01e','bash','-s'],input=script,text=True,capture_output=True,timeout=55)
def publish(receipt,name):
 marker=receipt.with_suffix('.published')
 if marker.exists():return
 subprocess.run([sys.executable,str(LOCAL/'round10_remote.py'),'put',str(receipt),REMOTE+'/transfer_receipts/'+name+'.json'],check=True)
 marker.write_text(str(time.time())+'\n')
def main():
 # Queue entries are explicit own paths published after local formal gates.
 while not (LOCAL/'ROUND14_STOP_TRANSFER').exists():
  tmp=LOCAL/'round14_transfer_queue.json'
  p=subprocess.run([sys.executable,str(LOCAL/'round10_remote.py'),'get',REMOTE+'/transfer_queue.json',str(tmp)],capture_output=True,text=True,timeout=60)
  if p.returncode:time.sleep(30);continue
  queue=json.loads(tmp.read_text())
  for item in queue['batches']:
   name=item['batch_id'];receipt=LOCAL/('round14_CCI_'+name+'_verified.json')
   if receipt.is_file():publish(receipt,name);continue
   if not name.replace('_','').isalnum() or not item['list_path'].startswith(REMOTE+'/'):raise ValueError('transfer entry outside own namespace')
   # This helper uses a bounded, forwarded SSH agent and the established
   # CCI-side lab48 SSH config; no global auth/config is changed.
   command=['bash','/home/neos/jzy/tools/cci_pull_from_48g.sh',item['list_path'],DEST+'/mirror']
   logfile=LOCAL/('round14_transfer_'+name+'.log')
   started=time.time()
   with logfile.open('a') as f:
    result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
   if result.returncode:raise RuntimeError('CCI pull failed, source copies retained: '+name)
   release=DEST+'/mirror'+item['release_path']
   script='''set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import json,hashlib
p=Path(RELEASE)
m=json.loads((p/'TRANSFER_SHA256.json').read_text());checks=[]
for row in m['files']:
 f=p/row['relative'];h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8<<20),b''):h.update(b)
 if f.stat().st_size!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('CCI SHA mismatch '+str(f))
 checks.append(row['relative'])
print(json.dumps({'status':'verified','release_directory':str(p),'file_count':len(checks),'bytes':sum(r['bytes'] for r in m['files'])}))
PY
'''.replace('RELEASE',repr(release))
   verified=sshcci(script)
   if verified.returncode:raise RuntimeError('CCI SHA/readback failed, source retained: '+verified.stderr)
   proof=json.loads(verified.stdout);proof.update(batch_id=name,wall_seconds=time.time()-started)
   receipt.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
   # The producer deletes its own rendered intermediates/local delivery only
   # after this receipt reaches its own run directory.
   publish(receipt,name)
   print(json.dumps(proof),flush=True)
  # Rebuild only our additive index; the existing 500-parent waiter is read-only.
  receipts={}
  for item in queue['batches']:
   receipt=LOCAL/('round14_CCI_'+item['batch_id']+'_verified.json')
   if receipt.is_file():receipts[item['batch_id']]=json.loads(receipt.read_text())
  if receipts:
   aggregate=(LOCAL/'round14_cci_aggregate.py').read_text()
   import base64
   payload=base64.b64encode(aggregate.encode()).decode()
   q=base64.b64encode(json.dumps(queue).encode()).decode()
   r=base64.b64encode(json.dumps(receipts).encode()).decode()
   script="set -euo pipefail\npython3 - <<'PY'\nimport base64,json\nnamespace={'__name__':'aggregate_readback'}\nexec(base64.b64decode("+repr(payload)+").decode(),namespace)\nvalue=namespace['main'](json.loads(base64.b64decode("+repr(q)+")),json.loads(base64.b64decode("+repr(r)+")),complete="+repr(bool(queue.get('complete')))+")\nprint(json.dumps({'status':value['status'],'formal_windows':value['formal_windows'],'formal_logical_questions':value['formal_logical_questions'],'independent_parent_scenes':value['independent_parent_scenes']}))\nPY\n"
   summary=sshcci(script)
   if summary.returncode:raise RuntimeError('own additive CCI index failed: '+summary.stderr)
   (LOCAL/'round14_CCI_aggregate_status.json').write_text(summary.stdout)
  if queue.get('complete'):return
  time.sleep(30)
if __name__=='__main__':main()
