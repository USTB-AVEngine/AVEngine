"""One genuinely fresh native process, scene upload and single-pose RIR."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.timeline.pilot_continuous_scenes import native_renderer
from avengine.timeline.v7_registered_motion_audio import position_at
p=argparse.ArgumentParser();p.add_argument('--motion',type=Path,required=True);p.add_argument('--house',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--time',type=float,required=True);p.add_argument('--quality',type=int,default=9);a=p.parse_args()
m=json.loads(a.motion.read_text());h=json.loads(a.house.read_text());inputs=json.loads(a.runtime.read_text());track=[{'time_s':f['time_s'],'position_m':f['mouth_world_m']} for f in m['frames']];point=position_at(track,a.time)
h['room']['placement']['camera_m']=m['listener_m'];h['room']['placement']['source_1_m']=point.tolist()
for k in ('direct_ray_count','indirect_ray_count','source_ray_count'):inputs['simulation'][k]*=a.quality
renderer,identity=native_renderer(inputs,h);batch=renderer.render([point.tolist()]);np.save(a.output,batch.samples[0],allow_pickle=False)
a.output.with_suffix('.json').write_text(json.dumps({'identity':identity,'position_m':point.tolist(),'cpu_s':batch.process_cpu_seconds,'setup':renderer.setup_report},indent=2))
