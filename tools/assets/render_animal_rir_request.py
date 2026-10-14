"""Render one exact requested point in a fresh native process."""
import argparse,json,time
from pathlib import Path
import numpy as np
from tools.timeline.pilot_continuous_scenes import native_renderer

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args();request=json.loads(a.request.read_text());start=time.process_time();house=request['house'];house['room']['placement']['source_1_m']=request['position_m']
 renderer,identity=native_renderer(request['inputs'],house);result=renderer.render([request['position_m']]);np.save(request['wave'],result.samples[0],allow_pickle=False)
 Path(request['receipt']).write_text(json.dumps({'position_m':request['position_m'],'identity':identity,'native_setup':renderer.setup_report,'process_cpu_seconds':time.process_time()-start},indent=2))
if __name__=='__main__':main()
