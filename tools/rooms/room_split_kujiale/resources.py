"""Observe only this task's process tree, using a thread in its parent."""
import os, resource, threading, time
import psutil
from tools.rooms.room_split_kujiale.adapter import dump


class Resources:
    def __init__(self):
        self.stop=threading.Event();self.rows=[];self.process=psutil.Process(os.getpid())
        self.thread=threading.Thread(target=self.observe,daemon=True)
    def observe(self):
        while not self.stop.is_set():
            processes=[self.process]+self.process.children(recursive=True)
            row=dict(time=time.time(),processes=[])
            for p in processes:
                try:
                    m=p.memory_info();lim=p.rlimit(resource.RLIMIT_AS)
                    row['processes'].append(dict(pid=p.pid,ppid=p.ppid(),nice=p.nice(),rss=m.rss,vms=m.vms,AS_soft_limit=lim[0],command=p.cmdline(),create_time=p.create_time()))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            self.rows.append(row);self.stop.wait(10)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()
    def save(self,path):
        dump(path,dict(samples=self.rows,interval_s=10,
             maximum_processes_sampled=max((len(r['processes']) for r in self.rows),default=0),
             maximum_rss_gib_sampled=max((sum(p['rss'] for p in r['processes'])/1024**3 for r in self.rows),default=0),
             maximum_vms_gib_sampled=max((sum(p['vms'] for p in r['processes'])/1024**3 for r in self.rows),default=0),
             scope='this parent and its descendants only; monitor runs in a thread; no external sampler stopped'))
