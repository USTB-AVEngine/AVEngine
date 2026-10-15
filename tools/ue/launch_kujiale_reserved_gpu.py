#!/usr/bin/env python3
"""Launch UE on the UUID-authorized card in an isolated device namespace."""
import argparse,os
from avengine.capture.kujiale_gpu_queue import launch_reserved_gpu
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--physical-gpu',type=int,required=True);a,rest=p.parse_known_args();launch_reserved_gpu(a.output_root,a.physical_gpu,rest)
