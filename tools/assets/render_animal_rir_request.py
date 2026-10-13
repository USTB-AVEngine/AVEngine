"""Render one exact requested point in a fresh native process."""
import argparse,hashlib,io,json,os,time
from pathlib import Path
import numpy as np
from tools.timeline.pilot_continuous_scenes import native_renderer

def save_fresh_wave(wave,request):
    output=Path(request["wave"])
    root=request.get("immutable_wave_root")
    if root is None:
        np.save(output,wave,allow_pickle=False)
        return {"immutable_pool_used":False}
    import fcntl
    root=Path(root).resolve()
    if not output.resolve().is_relative_to(root) or not Path(request["receipt"]).resolve().is_relative_to(root):
        raise ValueError("wave_pool_output_outside_owned_root")
    encoded=io.BytesIO();np.save(encoded,wave,allow_pickle=False);payload=encoded.getvalue()
    digest=hashlib.sha256(payload).hexdigest()
    pool=root/"cache"/"native_wave_pool";pool.mkdir(parents=True,exist_ok=True)
    canonical=pool/(digest+".npy")
    with (pool/(digest+".lock")).open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if canonical.exists():
            if canonical.read_bytes()!=payload:raise ValueError("fresh_native_wave_pool_hash_collision_or_mutation")
        else:
            with canonical.open("xb") as stored:stored.write(payload)
        os.link(canonical,output)
    return {"immutable_pool_used":True,"wave_sha256":digest,"pool":str(canonical),"fresh_native_render_completed_before_deduplication":True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args();request=json.loads(a.request.read_text());start=time.process_time();house=request['house'];house['room']['placement']['source_1_m']=request['position_m']
 renderer,identity=native_renderer(request['inputs'],house);result=renderer.render([request['position_m']]);storage=save_fresh_wave(result.samples[0],request)
 Path(request['receipt']).write_text(json.dumps({'position_m':request['position_m'],'identity':identity,'native_setup':renderer.setup_report,'process_cpu_seconds':time.process_time()-start,'native_process_pid':os.getpid(),'wave_storage':storage},indent=2))
if __name__=='__main__':main()
