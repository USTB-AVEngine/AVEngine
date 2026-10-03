#!/usr/bin/env python3
"""Measure accepted production receipts, CPU/GPU occupancy and retained disk.

CPU is measured user+system process time, not wall time times an assumed core
count. Native render wrapper monitoring overhead is unrecorded in older pilot
receipts, so its CPU total is a small lower bound. No answer files are opened.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

def read(path):return json.loads(Path(path).read_text())
def require(value,reason):
    if not value:raise ValueError(reason)
def disk(paths):
    seen=set();apparent=allocated=0;files=0
    for top in paths:
        for path in ([top] if top.is_file() else top.rglob('*')):
            if path.is_symlink() or not path.is_file():continue
            s=path.stat();inode=(s.st_dev,s.st_ino)
            if inode in seen:continue
            seen.add(inode);apparent+=s.st_size;allocated+=s.st_blocks*512;files+=1
    return {'unique_inode_bytes':apparent,'allocated_bytes':allocated,'unique_files':files}
def describe(values):return {'mean':statistics.mean(values),'min':min(values),'max':max(values),'sum':sum(values)}
def receipts(group,verification=None):
    rows=[]
    for label,path in [('prepare',group/'prepare_timing.json'),('audio',group/'audio_timing.json'),
                       ('capture_v0',group/'visual/v0/timing.json'),('capture_v1',group/'visual/v1/timing.json'),('certify',group/'certify_timing.json')]:
        if not path.exists():continue
        value=read(path);rows.append({'stage':label,'path':str(path),'wall_seconds':value['wall_seconds'],
          'cpu_seconds':value.get('cpu_seconds',0)+value.get('child_cpu_seconds',0),
          'gpu_peak_process_mib':value.get('gpu_peak_process_mib')})
    if verification:rows.append({'stage':'verify_once','path':verification['_path'],'wall_seconds':verification['wall_seconds'],
      'cpu_seconds':verification['cpu_seconds']+verification['child_cpu_seconds'],'gpu_peak_process_mib':None})
    return rows
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--verification',type=Path,required=True)
    p.add_argument('--expected-groups',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=a.root.resolve();checked=read(a.verification);require(checked['status']=='pass','verification not pass')
    verified={v['group']:{**v,'_path':str(a.verification)} for v in checked['groups']}
    certs=[read(path) for path in sorted((r/'scoring_private/certificates').glob('*.json'))]
    require(len(certs)==a.expected_groups and len(verified)==a.expected_groups,'incomplete panel')
    results=[];parents=[];public=[];support=[];c50=[];snr=[];foa_angle=[];scenarios=set()
    for c in certs:
        g=Path(c['provenance']['recipe']).parent;parents.append(g);recipe=read(g/'recipe.json');scenarios.add(recipe['scenario']['id'])
        rows=receipts(g,verified[c['group']]);require(len(rows)==6,'accepted production timing incomplete')
        for review in c['joint_solvability']['appearance']:
            for actor in review['actors'].values():
                for check in actor['checks']:
                    if 'candidate_counts' in check:support.append(check['candidate_counts'].get(str(check['observed_value']).casefold(),0))
        c50.extend(v['c50_db'] for v in c['joint_solvability']['rir_clarity'])
        snr.extend(v['target_to_competitor_db'] for v in c['joint_solvability']['target_stems'])
        foa_angle.extend(v['check']['angle_deg'] for v in c['joint_solvability']['foa_direction'])
        files=[r/'public/groups'/m['sample_id'] for m in c['media']['members']];public.extend(files)
        gpu=sum(v['wall_seconds'] for v in rows if v['stage'].startswith('capture'))
        peaks=[v['gpu_peak_process_mib'] for v in rows if v['gpu_peak_process_mib'] is not None]
        results.append({'group':c['group'],'house':c['house'],'room':c['room'],'quota':c['quota'],
          'wall_seconds_success_path':sum(v['wall_seconds'] for v in rows),'cpu_seconds_measured':sum(v['cpu_seconds'] for v in rows),
          'gpu_occupied_wall_seconds':gpu,'gpu_peak_process_mib':max(peaks) if peaks else None,
          'full_native_audit_disk':disk([g]),'public_media_disk':disk(files),'receipts':rows})
    rejected=[]
    for directory in sorted((r/'rejected').glob('*')):
        if not directory.is_dir():continue
        values=receipts(directory);rejection=read(directory/'rejection.json') if (directory/'rejection.json').exists() else {}
        rejected.append({'directory':str(directory),'known_stage_wall_seconds':sum(v['wall_seconds'] for v in values),
          'known_stage_cpu_seconds':sum(v['cpu_seconds'] for v in values),'missing_failed_stage_timing':True,'rejection':rejection,'disk':disk([directory])})
    wall=describe([v['wall_seconds_success_path'] for v in results]);cpu=describe([v['cpu_seconds_measured'] for v in results]);gpu=describe([v['gpu_occupied_wall_seconds'] for v in results])
    known_retry_wall=sum(v['known_stage_wall_seconds'] for v in rejected)
    count=len(results);ratio=(count+len(rejected))/count
    report={'status':'pass','groups':count,'members':4*count,'house_counts':dict(Counter(c['house'] for c in certs)),
      'quota_counts':dict(Counter(c['quota']['voice']+'/'+c['quota']['side'] for c in certs)),
      'room_counts':dict(Counter(c['house']+'/'+c['room'] for c in certs)),'dialogue_scenarios':len(scenarios),
      'wall_seconds_success_path':wall,'cpu_seconds_measured':cpu,'gpu_occupied_wall_seconds':gpu,
      'gpu_peak_process_mib':max(v['gpu_peak_process_mib'] for v in results if v['gpu_peak_process_mib'] is not None),
      'full_accepted_native_audit_disk':disk(parents),'public_media_disk':disk(public),'total_run_disk':disk([r]),
      'known_rejection_stage_wall_seconds':known_retry_wall,'known_rejection_stage_cpu_seconds':sum(v['known_stage_cpu_seconds'] for v in rejected),
      'whole_parent_rejections':len(rejected),'attempt_to_accepted_ratio':ratio,
      'measured_quality':{'min_color_support_pixels':min(support) if support else 'see certificates',
         'min_rir_c50_db':min(c50),'min_target_to_competitor_db':min(snr),'max_foa_direct_angle_error_deg':max(foa_angle)},
      'extrapolation_720':{'baseline_success_path_hours':wall['mean']*720/3600,
        'known_retries_lower_bound_hours':(wall['sum']+known_retry_wall)/count*720/3600,
        'retry_all_full_cost_planning_hours':wall['mean']*ratio*720/3600,
        'baseline_four_assigned_gpu_workers_hours':wall['mean']*720/3600/4,
        'retry_all_full_cost_four_workers_hours':wall['mean']*ratio*720/3600/4,
        'cpu_core_hours_measured':cpu['mean']*720/3600,'gpu_occupied_hours':gpu['mean']*720/3600,
        'native_audit_bytes':disk(parents)['allocated_bytes']/count*720,'public_media_bytes':disk(public)['allocated_bytes']/count*720,
        'hardware':'48g measurements extrapolated; CCI speed and remaining ten houses unverified'},
      'measurement_notes':['CPU uses recorded user+system process times and child CPU, not CPU wall minutes.',
        'Older pilot native capture receipts omit producer wrapper CPU; failed stages may have no receipt. CPU/retry sums are lower bounds.',
        'Success path counts one independent verification per group; repeated audits, software setup, SSH waits and review time are excluded.',
        'GPU minutes are occupied native rendering wall time, not utilization-adjusted kernel time; memory is per-owned-PID sampled at 0.5 s.',
        'Disk deduplicates hardlinks by device/inode; includes actual file allocation, not four copies of shared member media.',
        'Rejected parents and prototypes are retained and reported separately; no deletion or model selection.'],
      'verification':str(a.verification),'group_results':results,'rejected_results':rejected,'answers_opened':False,'models_run':False}
    with a.output.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'group_results','rejected_results'}},ensure_ascii=False))
if __name__=='__main__':main()
