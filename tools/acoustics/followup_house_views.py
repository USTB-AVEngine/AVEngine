#!/usr/bin/env python3
"""CPU visual checks as this follow-up's owned native house files complete."""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.train_house_followup import read,save

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    plan=[]
    for name,root in [('hm3d_manifest.json','hm3d_current'),('hm3d_blocked_manifest.json','hm3d_blocked')]:
        manifest=a.work/name
        for h in read(manifest)['houses']:plan.append((h['house'],manifest,a.work/root))
    done=set();futures={};rows=[]
    def one(house,manifest,root):
        d=root/'houses'/house;result=read(d/'result.json')
        if not result.get('foa_checks'):
            value={'status':'review','reasons':['NATIVE_ACOUSTIC_INPUT_OR_MEASUREMENT_INCOMPLETE; visual projection needs measured FOA probes'],'compute_device':'CPU','measurement_run':False}
            save(d/'visual_checks.json',value);return {'house':house,'visual_status':'review','native_probes_missing':True}
        log=a.work/'logs'/('visual_'+house+'.log')
        command=[sys.executable,str(REPO/'tools/acoustics/accept_houses.py'),'visual','--manifest',str(manifest),'--output',str(root),'--house',house]
        with log.open('x') as f:
            proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            save(a.work/'evidence'/('visual_pid_'+house+'.json'),dict(pid=proc.pid,argv=command,log=str(log)))
            try:rc=proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                rc=-1
        if not (d/'visual_checks.json').exists():save(d/'visual_checks.json',dict(status='review',reasons=['CPU_VISUAL_RUN_FAILED; original geometry retained'],returncode=rc,log=str(log),compute_device='CPU'))
        return dict(house=house,visual_status=read(d/'visual_checks.json')['status'],returncode=rc,log=str(log))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        while len(done)<len(plan):
            for item in plan:
                house,manifest,root=item
                if house in done or house in futures:continue
                path=root/'houses'/house
                if (path/'visual_checks.json').exists():done.add(house);continue
                if not (path/'result.json').exists():continue
                try:read(path/'result.json')
                except json.JSONDecodeError:continue
                futures[house]=pool.submit(one,*item)
            for house,future in list(futures.items()):
                if future.done():
                    try:r=future.result()
                    except Exception as exc:r=dict(house=house,visual_status='review',error=repr(exc))
                    rows.append(r);done.add(house);del futures[house];print('HM3D_VISUAL',json.dumps(r),flush=True)
            log=a.work/'logs/native_driver.log'
            if log.exists() and 'ALL_HM3D_NATIVE_FINISHED' in log.read_text() and not futures:
                remaining=[h for h,m,r in plan if h not in done]
                if remaining:raise RuntimeError('Missing native results after driver completion: '+repr(remaining))
            if len(done)<len(plan):time.sleep(3)
    save(a.work/'evidence/hm3d_visual_completion.json',dict(total_houses=len(plan),completed=len(done),new_visual_runs=rows,compute_device='CPU'))
    print('ALL_HM3D_VISUAL_FINISHED',len(done),flush=True)
if __name__=='__main__':main()
