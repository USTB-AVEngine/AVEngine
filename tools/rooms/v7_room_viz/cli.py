"""Build the v7 review page and supervise only task-owned native renderers."""
from __future__ import annotations
import argparse
import json
import os
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from .common import now, prepare_jobs, read_json, write_json


def gpu_sample(gpu, pid=None):
    result = subprocess.check_output(['nvidia-smi','-q','-x','-i',str(gpu)],text=True,timeout=15)
    device = ET.fromstring(result).find('gpu')
    def mib(text):
        try:
            return int(float(text.split()[0]))
        except (ValueError,AttributeError):
            return 0
    processes = [dict(pid=int(p.findtext('pid')),used_mib=mib(p.findtext('used_memory')),
                      type=p.findtext('type'),name=p.findtext('process_name'))
                 for p in device.findall('processes/process_info')]
    return dict(at=now(),gpu=gpu,uuid=device.findtext('uuid'),
                free_mib=mib(device.findtext('fb_memory_usage/free')),
                our_used_mib=sum(p['used_mib'] for p in processes if p['pid']==pid),
                processes=processes)


def supervise(args):
    jobs = sorted((args.output/'jobs').glob('*.json'))
    if args.house:
        jobs = [p for p in jobs if p.stem == args.house]
    else:
        recommended = read_json(args.input/'report_final/recommended_houses.json')['houses']
        ranks = {x['house']:i for i,x in enumerate(recommended)}
        jobs.sort(key=lambda p: (ranks.get(p.stem,99), p.stem))
    if not jobs:
        raise ValueError('no prepared jobs for the requested houses')
    record = args.output/'evidence'/(('render_'+args.house) if args.house else 'render_batch')
    receipt = dict(started_at_sgt=now(),supervisor_pid=os.getpid(),status='running',
                   gpu_policy='free >= prior sampled peak + 4096 MiB; initial conservative 8192 MiB peak',
                   measured_peak_mib=0,houses=[],current_renderer=None)
    record_path = record.with_suffix('.json')
    write_json(record_path,receipt)
    start = time.monotonic()
    peak = args.prior_peak_mib
    for job in jobs:
        dest = args.output/'rendered_houses'/(job.stem+'.json')
        if dest.exists() and read_json(dest).get('status')=='complete' and not args.refresh:
            continue
        required = peak + 4096
        while True:
            if time.monotonic()-start > args.timeout:
                receipt.update(status='timeout_waiting_for_headroom',at_sgt=now())
                write_json(record_path,receipt)
                return 2
            samples = [gpu_sample(gpu) for gpu in args.gpus]
            chosen = max(samples,key=lambda x:x['free_mib'])
            if chosen['free_mib'] >= required:
                break
            receipt.update(waiting_for_headroom=dict(required_free_mib=required,samples=samples))
            write_json(record_path,receipt)
            print('WAITING_HEADROOM',job.stem,required,chosen['free_mib'],flush=True)
            time.sleep(15)
        command = [sys.executable,'-u','-m','tools.rooms.v7_room_viz.cli','worker',
                   '--input',str(args.input),'--output',str(args.output),'--house',job.stem]
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(chosen['gpu']),HABITAT_SIM_LOG='quiet',MAGNUM_LOG='quiet',
                   CUDA_CACHE_PATH=str(args.output/'cache/cuda'),
                   __GL_SHADER_DISK_CACHE_PATH=str(args.output/'cache/gl'),
                   XDG_CACHE_HOME=str(args.output/'cache/xdg'))
        logpath = args.output/'logs'/(job.stem+('.refresh.log' if args.refresh else '.log'))
        with logpath.open('a') as log, record.with_suffix('.vram.jsonl').open('a') as samples_file:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            item = dict(house=job.stem,pid=child.pid,gpu=chosen['gpu'],command=command,log=str(logpath),
                        launch=chosen,required_free_mib=required,measured_peak_mib=0)
            receipt['current_renderer']=item
            write_json(record_path,receipt)
            print('LAUNCHED',job.stem,child.pid,'GPU',chosen['gpu'],'free',chosen['free_mib'],'required',required,flush=True)
            child_start = time.monotonic()
            failure = None
            while child.poll() is None:
                sample = gpu_sample(chosen['gpu'],child.pid)
                samples_file.write(json.dumps(sample)+'\n')
                samples_file.flush()
                item['measured_peak_mib']=max(item['measured_peak_mib'],sample['our_used_mib'])
                item['minimum_free_mib']=min(item.get('minimum_free_mib',sample['free_mib']),sample['free_mib'])
                if sample['free_mib']<4096:
                    failure='shared GPU free memory fell below the 4 GiB reserve'
                if time.monotonic()-child_start>1800 or time.monotonic()-start>args.timeout:
                    failure='task-owned native renderer exceeded timeout'
                if failure:
                    # Popen identifies the exact child created above. No unrelated PID is touched.
                    child.terminate()
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
                write_json(record_path,receipt)
                time.sleep(.5)
            item.update(returncode=child.wait(),failure=failure,finished_at_sgt=now())
            peak=max(peak,item['measured_peak_mib'])
            receipt['measured_peak_mib']=max(receipt['measured_peak_mib'],item['measured_peak_mib'])
            receipt['houses'].append(item)
            receipt['current_renderer']=None
            write_json(record_path,receipt)
            print('FINISHED',job.stem,item['returncode'],'peak_MiB',item['measured_peak_mib'],flush=True)
            if failure:
                receipt.update(status='stopped_owned_renderer',finished_at_sgt=now())
                write_json(record_path,receipt)
                return 2
    receipt.update(status='complete',finished_at_sgt=now(),seconds=time.monotonic()-start,
                   failed_houses=[x['house'] for x in receipt['houses'] if x['returncode']])
    write_json(record_path,receipt)
    return 1 if receipt['failed_houses'] else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','render','worker','build'])
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--house')
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    parser.add_argument('--prior-peak-mib',type=int,default=8192)
    parser.add_argument('--timeout',type=int,default=10800)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    args.input=args.input.resolve(strict=True)
    args.output=args.output.resolve()
    if args.input==args.output or args.input in args.output.parents:
        raise ValueError('visualization output must not be inside the read-only input root')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if args.action=='prepare':
        prepare_jobs(args.input,args.output)
    elif args.action=='worker':
        if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
            raise ValueError('worker requires exactly one explicitly selected visible GPU')
        from .render import render_house
        render_house(args.output/'jobs'/(args.house+'.json'),args.output)
    elif args.action=='render':
        return supervise(args)
    elif args.action=='build':
        from .page import build
        build(args.input,args.output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
