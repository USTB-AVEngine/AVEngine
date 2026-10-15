#!/usr/bin/env python3
"""Publish each newly passed room registration and append READY_TO_MERGE."""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
from tools.registry.register_kujiale_ue_measurements import load,publish

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--registration',type=Path,required=True);p.add_argument('--stop-file',type=Path,required=True);p.add_argument('--poll-seconds',type=float,default=10.);a=p.parse_args();m=load(a.manifest);root=Path(m['output_root']);work=Path(m['worktree']);state=root/'evidence/room_ready_receipts.json';seen=load(state)if state.exists()else{r['house']+'/'+r['room_id']:{'initial_pass':True}for r in load(a.registration)['rooms']if r['smoke_status']=='pass'}
    while not a.stop_file.exists():
        try:
            m=load(a.manifest);data=publish(m,a.registration);new=[r for r in data['rooms']if r['smoke_status']=='pass'and r['house']+'/'+r['room_id']not in seen]
            for room in new:
                subprocess.run(['git','add','--',str(a.registration),str(a.registration)+'.sha256'],cwd=work,check=True)
                if subprocess.run(['git','diff','--cached','--quiet','--',str(a.registration),str(a.registration)+'.sha256'],cwd=work).returncode:
                    subprocess.run(['git','commit','--only',str(a.registration),str(a.registration)+'.sha256','-m','registry: qualify native Kujiale '+room['house']+'/'+room['room_id']],cwd=work,check=True)
                head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip();key=room['house']+'/'+room['room_id'];seen[key]={'commit':head,'independent_questions':room['independent_questions'],'scene_id':room['scene_id']}
                with(root/'PROGRESS.md').open('a')as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' READY_TO_MERGE '+head+' G '+key+' 当前UE实测矩阵/原始加载包/正式登记+完整14题/5绑定/五窗去重/英文/reader通过，scene='+room['scene_id']+' independent='+str(room['independent_questions'])+' family='+json.dumps(room['families'],ensure_ascii=False)+'；原始登记不改，见ALL_ROOMS_zh.md及正式登记。\n')
            state.write_text(json.dumps(seen,ensure_ascii=False,indent=2)+'\n');subprocess.run([os.sys.executable,str(root/'scripts/write_fix_status.py')],cwd=work,check=False)
        except Exception as error:
            with(root/'PROGRESS.md').open('a')as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' G room handoff observation error '+repr(error)+'；不停止任何计算。\n')
        time.sleep(max(2.,a.poll_seconds))
if __name__=='__main__':main()
