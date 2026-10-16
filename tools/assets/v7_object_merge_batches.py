"""Compatibility entry point: collect all declared plans from per-asset receipts."""
import argparse
from pathlib import Path
from v7_object_collect import main as collect_main
import sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task-root',type=Path,required=True)
    p.add_argument('--worktree',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--prefix',default='pixal_partition',help='Legacy argument; per-asset receipts are authoritative')
    p.add_argument('--count',type=int,default=2,help='Legacy argument; all declared plans are collected')
    a=p.parse_args()
    sys.argv=[str(Path(__file__)),'--task-root',str(a.task_root),'--worktree',str(a.worktree)]
    collect_main()
if __name__=='__main__':main()
