"""Plot recorded P19 cgroup counters separately from host /proc statistics."""
import argparse,csv,datetime,hashlib,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('use a new immutable plot output')
    a.output.mkdir(parents=True);table=[];segments=[];provenance=[]
    zone=datetime.timezone(datetime.timedelta(hours=8))
    for source in a.inputs:
        rows=[];invalid=0
        for line in source.read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:invalid+=1
        before=None;segment=[]
        for r in rows:
            t=datetime.datetime.fromisoformat(r['utc'].replace('Z','+00:00'));cores=None
            stat=dict(line.split()for line in r.get('cgroup',{}).get('cpu.stat','').splitlines())
            if before and before['hostname']==r['hostname']and before['pid']==r['pid']and 'usage_usec'in stat:
                dt=(t-datetime.datetime.fromisoformat(before['utc'].replace('Z','+00:00'))).total_seconds()
                old=dict(line.split()for line in before['cgroup']['cpu.stat'].splitlines())
                if dt>0:cores=(int(stat['usage_usec'])-int(old['usage_usec']))/1e6/dt
            loads=[float(x)for x in r['proc_loadavg'].split()[:3]]
            item={'UTC':r['utc'],'Singapore_time':t.astimezone(zone).isoformat(),'source':source.name,
                'cgroup_busy_cores_including_training':cores,'host_aggregate_busy_cores':r.get('aggregate_busy_core_equivalent'),
                'load_1m':loads[0],'load_5m':loads[1],'load_15m':loads[2],
                'workers_limit':r['workers_limit'],'active_parents':r['active_parents'],'P19_quota':r['cgroup'].get('cpu.max')}
            table.append(item);segment.append(item);before=r
        segments.append(segment);provenance.append({'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'valid_rows':len(rows),'incomplete_or_invalid_rows':invalid})
    if not table:raise ValueError('no actual CPU records')
    with (a.output/'CPU_curve.csv').open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for segment in segments:
        if not segment:continue
        ts=[datetime.datetime.fromisoformat(r['Singapore_time'])for r in segment];label=segment[0]['source'].split('_')[1]
        axes[0].plot(ts,[r['cgroup_busy_cores_including_training']for r in segment],label=label+' P19 cgroup incl. training')
        axes[0].plot(ts,[r['host_aggregate_busy_cores']for r in segment],linestyle=':',alpha=.6,label=label+' host /proc aggregate')
        axes[1].plot(ts,[r['workers_limit']for r in segment],linestyle='--',label=label+' dispatch limit')
        axes[1].plot(ts,[r['active_parents']for r in segment],label=label+' active parents')
        axes[2].plot(ts,[r['load_1m']for r in segment],label=label+' host load 1m')
    axes[0].axhline(88,color='black',linestyle='--',label='P19 cgroup quota: 88 cores')
    axes[0].set_ylabel('Busy core equivalents');axes[1].set_ylabel('Parent processes');axes[2].set_ylabel('Host load');axes[2].set_xlabel('Singapore time (UTC+8), 2026-10-05')
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=zone))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Recorded CPU use: P19 cgroup includes training and generation; host counters have wider scope')
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(a.output/'CPU_curve.png',dpi=150);plt.close(fig)
    vals=[r['cgroup_busy_cores_including_training']for r in table if r['cgroup_busy_cores_including_training']is not None]
    result={'status':'actual_counters_plotted','sources':provenance,'rows':len(table),'cgroup_busy_cores_including_training':{'n':len(vals),'median':statistics.median(vals),'max':max(vals)},'no_claim_of_generation_only_CPU':True,'CSV':str(a.output/'CPU_curve.csv'),'PNG':str(a.output/'CPU_curve.png')}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
