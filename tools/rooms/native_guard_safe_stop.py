#!/usr/bin/env python3
"""Stop one recorded task guard after verifying PID creation time and argv."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--created',type=float,required=True);p.add_argument('--record',type=Path,required=True)
    a=p.parse_args();process=psutil.Process(a.pid)
    cmd=process.cmdline()
    if process.create_time()!=a.created or '--request' not in cmd or str(a.request) not in cmd or not any(s.endswith('/v7_kujiale_ue_guard.py') for s in cmd):
        raise ValueError('guard identity mismatch; no signal sent')
    if a.record.exists():raise FileExistsError(a.record)
    a.record.write_text(json.dumps({'pid':a.pid,'create_time':a.created,'command':cmd,'signal':'SIGTERM','time_unix':time.time()},indent=2)+'\n')
    process.terminate()


if __name__=='__main__':main()
