"""Append recoverable owned-task milestones without rewriting previous progress."""
import argparse,datetime,fcntl
from pathlib import Path

def append(root,message,artifacts,next_step):
 root=Path(root);stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds');line='- '+stamp+' | '+message.replace('\n',' ')+' | 产物：'+artifacts.replace('\n',' ')+' | 下一步：'+next_step.replace('\n',' ')+'\n'
 with (root/'PROGRESS.md').open('a') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX);handle.write(line);handle.flush()

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--message',required=True);p.add_argument('--artifacts',required=True);p.add_argument('--next',dest='next_step',required=True);a=p.parse_args();append(a.task_root,a.message,a.artifacts,a.next_step)
if __name__=='__main__':main()
