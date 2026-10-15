#!/usr/bin/env python3
"""Sample task PSS outside bwrap so every private UE namespace sees all workers."""
import argparse,json,time
from pathlib import Path
from avengine.timeline.v7_kujiale_host_resources import sample_host
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--stop-file',type=Path,required=True);a=p.parse_args();peak=0
    while not a.stop_file.exists():
        value=sample_host(a.output_root);peak=max(peak,value['PSS_bytes']);path=a.output_root/'evidence/host_resource_peak.json';path.write_text(json.dumps({'peak_PSS_bytes':peak,'latest_sample':value['time_utc'],'measurement':'host procfs owned process trees; resources pause new requests only'},indent=2)+'\n');time.sleep(2)
