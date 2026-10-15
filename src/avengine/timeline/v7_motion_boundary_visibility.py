"""Independent native stage-ray proof for a brief occlusion-boundary interval."""
from pathlib import Path
import json,subprocess,sys
import numpy as np

def measure_boundary(cache,scene,trajectory,lo,hi,attempt,endpoint,sr=16000):
    from avengine.timeline.continuous_scene import position_at
    renderer=cache.renderer
    if not hasattr(renderer,'house')or not hasattr(renderer,'inputs'):
        return {'status':'unverified','direct_visibility_flips':False,'reason':'native house/runtime not available'}
    folder=Path(attempt)/(endpoint+'_boundary_visibility')/(str(lo)+'_'+str(hi));folder.mkdir(parents=True,exist_ok=True);request=folder/'request.json';report=folder/'result.json'
    a=max(0,lo-round(.05*sr));b=min(round(float(trajectory[-1]['time_s'])*sr),hi+round(.05*sr))
    samples=sorted({lo,hi,(lo+hi)//2,*range(a,b+1,max(1,round(.001*sr)))})
    value={'house':renderer.house,'inputs':renderer.inputs,'scene':scene,'sample_indices':samples,
        'positions_m':[position_at(trajectory,k/sr).tolist()for k in samples],
        'output':str(report),'maximum_boundary_neighbourhood_each_side_ms':50.,'source':'actual 30Hz native attached emitter trajectory interpolated on the emission clock'}
    if not report.exists():
        request.write_text(json.dumps(value))
        with(folder/'native.log').open('ab')as stream:
            process=subprocess.run([sys.executable,'-B','-m',__name__,'--request',str(request)],stdout=stream,stderr=subprocess.STDOUT)
        if process.returncode:raise ValueError('native boundary measurement failed: '+str(folder/'native.log'))
    result=json.loads(report.read_text());result['native_ray_report']=str(report);return result

def worker(path):
    from avengine.timeline.v7_native_resource_slots import acquire_native_resource
    lease=acquire_native_resource('rir')
    from avengine.capture.continuous_native_video import NativeContinuousVideo
    data=json.loads(Path(path).read_text());video=NativeContinuousVideo(data['house'],data['scene'],data['inputs'],[],source_material_cache_directory=data['inputs'].get('source_material_cache_directory'))
    rows=[];listener=np.asarray(data['scene']['listener_m'])
    try:
        for index,point in zip(data['sample_indices'],data['positions_m']):
            delta=np.asarray(point)-listener;distance=float(np.linalg.norm(delta));ray=video.hs.geo.Ray(video.mn.Vector3(listener),video.mn.Vector3(delta/distance));hits=video.sim.cast_ray(ray,max(0,distance-.005)).hits
            rows.append({'sample':index,'direct_visible':not bool(hits),'native_blocker_object_ids':sorted({int(h.object_id)for h in hits})})
    finally:video.close()
    result={'status':'measured','direct_visibility_flips':len({r['direct_visible']for r in rows})>1,'native_rays':rows,'neighbourhood_ms_each_side':50.,'sample_spacing_ms':1.,'native_physical_stage':data['house']['input'].get('visual_source'),'directions_for_QA_remain_geometric':True}
    Path(data['output']).write_text(json.dumps(result,indent=2)+'\n')
    if lease:lease.close()
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);worker(parser.parse_args().request)
