"""Render one exact requested point in a fresh native process."""
import argparse,json,time,os
from pathlib import Path
import numpy as np
from tools.timeline.pilot_continuous_scenes import native_renderer

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args();request=json.loads(a.request.read_text());start=time.process_time();house=request['house'];house['room']['placement']['source_1_m']=request['position_m']
 # Scope the remapping to this task's own requests; shared environments are untouched.
 plan_path=Path("/data/jzy/tmp/claude_hardbank_20260923/v7_rig_dogs_20261014/inputs/cpu_plan.json")
 if a.request.resolve().is_relative_to(plan_path.parents[1]) and plan_path.exists():
  cpu_plan=json.loads(plan_path.read_text());current=list(os.sched_getaffinity(0))
  mapped=cpu_plan["audio_core_map"].get(str(current[0])) if len(current)==1 else None
  if mapped is not None:os.sched_setaffinity(0,{int(mapped)})
 renderer,identity=native_renderer(request['inputs'],house);result=renderer.render([request['position_m']]);np.save(request['wave'],result.samples[0],allow_pickle=False)
 Path(request['receipt']).write_text(json.dumps({'position_m':request['position_m'],'cpu_affinity':sorted(os.sched_getaffinity(0)),'identity':identity,'native_setup':renderer.setup_report,'process_cpu_seconds':time.process_time()-start},indent=2))
if __name__=='__main__':main()
