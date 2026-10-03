#!/usr/bin/env python3
"""CPU-only restartable continuous-scene pilot and native RIR throughput probe."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from avengine.timeline.continuous_pilot import (
    PositionRIRCache, FormantSpeech, CPUProxyVideo, make_scene, validate_navigation,
    validate_interpolation, measure_scene, write_json,
)
from avengine.timeline.continuous_scene import SAMPLE_RATE, camera_basis, position_at, interpolated_filter, angular_distance
from avengine.acoustics.house_acceptance_metrics import foa_direction


def native_renderer(inputs, house, threads=1):
    from avengine.acoustics.runtime import RLRSimulationConfig, load_compiled_acoustic_scene
    from avengine.acoustics.rir_cache import _NativeRIRBatchRenderer
    scene = load_compiled_acoustic_scene(house["acoustic_package_manifest"], allow_nonpassing_research_qa=True)
    place = house["room"]["placement"]
    sim = replace(RLRSimulationConfig.from_mapping(inputs["simulation"]), thread_count=threads)
    renderer = _NativeRIRBatchRenderer(scene, sim, batch_size=1, initial_positions_m=[place["source_1_m"]],
                                      listener_position_m=place["camera_m"], listener_orientation_wxyz=[1.,0.,0.,0.],
                                      layout_type="ambisonics", channel_count=4, hrtf_file_path="",
                                      source_radius_m=inputs["source_radius_m"], listener_radius_m=inputs["listener_radius_m"], **inputs["runtime"])
    identity = {"package_id":scene.package_id,"existing_package_content_sha256":scene.package_content_sha256,
                "package_manifest":str(scene.manifest_path),"simulation":{**inputs["simulation"],"thread_count":threads},
                "listener_position_m":place["camera_m"],"listener_orientation_wxyz":[1.,0.,0.,0.],
                "source_radius_m":inputs["source_radius_m"],"listener_radius_m":inputs["listener_radius_m"],
                "runtime_identity":renderer.setup_report["runtime"]}
    return renderer, identity


def house_worker(task):
    inputs_path, output_path, house_id, limit = task
    inputs = json.loads(Path(inputs_path).read_text())
    house = next(h for h in inputs["houses"] if h["house"] == house_id)
    output = Path(output_path)/"houses"/house_id
    output.mkdir(parents=True, exist_ok=True)
    worker_start, cpu_start = time.perf_counter(), time.process_time()
    renderer, identity = native_renderer(inputs, house)
    write_json(output/f"native_setup_{time.time_ns()}.json",renderer.setup_report)
    print("NATIVE_READY",house_id,renderer.setup_report["wall_seconds"],flush=True)
    cache = PositionRIRCache(output/"rir_cache",renderer,identity)
    validation_cache = PositionRIRCache(output/"validation_rir_cache",renderer,identity)
    tts = FormantSpeech()
    first_scene = make_scene(house,0,30.)
    # 200 ms starts align both 100 ms annotation and 15 fps frame grids exactly.
    def align(scene):
        scene["window_starts_s"] = [round(t*5)/5 for t in scene["window_starts_s"]]
        return scene
    first_scene = align(first_scene)
    from avengine.acoustics.runtime import load_habitat_runtime
    hs, _ = load_habitat_runtime(runtime_mode="current-installed", **inputs["runtime"])
    geometry_root=output/"geometry";geometry_root.mkdir(exist_ok=True)
    if not (geometry_root/"navigation.json").exists():
        nav = [validate_navigation(house,make_scene(house,i,30.),hs) for i in range(3)]
        write_json(geometry_root/"navigation.json",nav)
    stamp = time.perf_counter()
    print("NAV_READY",house_id,flush=True)
    video = CPUProxyVideo(house,first_scene,inputs,geometry_root)
    print("CPU_BACKGROUND_READY",house_id,time.perf_counter()-stamp,flush=True)
    write_json(geometry_root/f"cpu_visual_setup_{time.time_ns()}.json",{**video.receipt,"wall_seconds":time.perf_counter()-stamp})
    receipts = []
    count = 0
    for duration in (30.,60.):
        for index in range(3):
            if limit and count >= limit:
                break
            scene=align(make_scene(house,index,duration))
            validation_root = output/f"validation_s{index:02d}"
            validation_root.mkdir(exist_ok=True)
            path=validation_root/"interpolation_validation.json"
            if not path.exists():
                print("VALIDATE_START",house_id,index,flush=True)
                validation=validate_interpolation(scene,validation_cache,validation_root)
                print("VALIDATE_DONE",house_id,index,validation["policies"]["selected"]["max_foa_error_deg"],flush=True)
            else:
                validation=json.loads(path.read_text())
            if not validation["policies"]["selected"]["pass_5deg"]:
                raise ValueError("selected RIR update policy exceeds 5 degree direction error")
            print("SCENE_START",scene["scene_id"],flush=True)
            receipt=measure_scene(scene,house,cache,tts,video,output/"scenes"/scene["scene_id"])
            receipts.append(receipt)
            count += 1
            print("SCENE_DONE",house_id,scene['scene_id'],receipt['wall_seconds'],receipt['total_cpu_seconds'],flush=True)
    result={"house":house_id,"pid":os.getpid(),"compute_device":"CPU","worker_wall_seconds":time.perf_counter()-worker_start,
            "worker_process_cpu_seconds":time.process_time()-cpu_start,"scene_count":len(receipts),"cache":cache.stats,"validation_cache":validation_cache.stats,
            "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    write_json(output/f"worker_receipt_{time.time_ns()}.json",result)
    return result


def bench_worker(task):
    inputs_path, house_index, count, job_index = task
    inputs = json.loads(Path(inputs_path).read_text());house=inputs["houses"][house_index]
    before=time.time(); cpu=time.process_time()
    renderer, _ = native_renderer(inputs,house)
    setup_done=time.time()
    p=house["room"]["placement"]
    a,b=np.array(p["source_1_m"]),np.array(p["source_2_m"])
    renderer.render([a.tolist()])
    warmup_done=time.time(); records=[]
    for i in range(count):
        point=a+((i+1)/(count+2))*(b-a)
        batch=renderer.render([point.tolist()])
        records.append({"wall_seconds":batch.wall_seconds,"cpu_seconds":batch.process_cpu_seconds,"sample_count":batch.samples[0].shape[1]})
    return {"job":job_index,"house":house["house"],"pid":os.getpid(),"start_epoch":before,"setup_done_epoch":setup_done,
            "simulation_start_epoch":warmup_done,"end_epoch":time.time(),"cpu_seconds":time.process_time()-cpu,
            "rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"records":records}


def benchmark(args):
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    result=[]
    for workers in args.workers:
        if not 1 <= workers <= 64:
            raise ValueError("48g permits at most 64 native working processes")
        start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            rows=list(pool.map(bench_worker,[(str(args.inputs),i%2,args.rirs_per_worker,i) for i in range(workers)]))
        wall=time.perf_counter()-start
        native_cpu=sum(x["cpu_seconds"] for x in rows)
        sim_wall=sum(v["wall_seconds"] for x in rows for v in x["records"])
        sim_cpu=sum(v["cpu_seconds"] for x in rows for v in x["records"])
        span=max(x["end_epoch"] for x in rows)-min(x["simulation_start_epoch"] for x in rows)
        record={"workers":workers,"rlr_threads_per_worker":1,"measured_rirs":workers*args.rirs_per_worker,
                "controller_wall_seconds":wall,"sum_worker_cpu_seconds":native_cpu,
                "native_simulation_sum_wall_seconds":sim_wall,"native_simulation_sum_cpu_seconds":sim_cpu,
                "simulation_span_seconds":span,"rirs_per_simulation_span_second":workers*args.rirs_per_worker/span,
                "simultaneous_memory_estimate_bytes":sum(x["rss_bytes"] for x in rows),"jobs":rows}
        write_json(output/f"workers_{workers:02d}.json",record);result.append(record)
        print("BENCH_DONE",workers,wall,sim_cpu,flush=True)
    write_json(output/"summary.json",result)


def review_windows(args):
    """Five fixed window IDs; read encoded pixels and independent native FOA."""
    from PIL import Image,ImageDraw
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    inputs=json.loads(args.inputs.read_text())
    chosen=[(0,0,0),(0,1,1),(0,2,2),(1,0,0),(1,1,1)]
    reports=[]
    for house_index,index,window_index in chosen:
        house=inputs["houses"][house_index]
        house_root=args.pilot_root/"houses"/house['house']
        scene_id=f"{house['house']}__s{index:02d}__30s"
        scene_root=house_root/"scenes"/scene_id
        receipt=json.loads((scene_root/"completed.json").read_text());scene=receipt['scene']
        attempt=scene_root/receipt['attempt'];window=receipt['windows'][window_index]
        folder=scene_root/window['path'];window_start=window['start_s']
        voiced=[e for e in scene['events'] if e['endpoint_id']=='voice2' and e['start_tick']/10 < window_start+10 and e['end_tick']/10 > window_start+.2]
        event=voiced[0]
        query=max(window_start+.2,event['start_tick']/10+.2)
        frame=int(np.ceil(query*15));query=frame/15
        if query >= min(window_start+10,event['end_tick']/10):
            raise ValueError('fixed window has no valid active query')
        local_frame=frame-round(window_start*15)
        payload=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-threads','1','-i',str(folder/'video.mp4'),
                                '-vf',f'select=eq(n\\,{local_frame})','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],check=True,capture_output=True).stdout
        rgb=np.frombuffer(payload,np.uint8).reshape(720,1280,3).copy()
        listener=np.array(scene['listener_m']);position=position_at(scene['sources']['voice2']['trajectory'],query)
        f,r,up=camera_basis(scene['camera_forward']);relative=position-listener
        focal=1280/(2*np.tan(np.radians(85/2)))
        expected=np.array([640+focal*(relative@r)/(relative@f),360-focal*(relative@up)/(relative@f)])
        yy,xx=np.indices((720,1280))
        roi=(abs(xx+.5-expected[0]) < 70)&(abs(yy+.5-expected[1]) < 70)
        distance=np.linalg.norm(rgb.astype(float)-[245,110,35],axis=2)
        mask=roi&(distance < 42)
        ids=np.argwhere(mask)
        if len(ids)<12:
            raise ValueError('encoded source head marker not independently visible')
        centroid=np.array([float(np.mean(ids[:,1]+.5)),float(np.mean(ids[:,0]+.5))])
        pixel_ray=f+(centroid[0]-640)/focal*r+(360-centroid[1])/focal*up
        renderer,identity=native_renderer(inputs,house)
        cache=PositionRIRCache(house_root/'rir_cache',renderer,identity)
        native=cache.get(position)
        native_foa=foa_direction(native,listener,position,f.copy())['check']
        rir_plan=json.loads((attempt/'rir_plan.json').read_text())['voice2']['keyframes']
        filters=[cache.get(k['position_m']) for k in rir_plan]
        blend=interpolated_filter(filters,[k['sample'] for k in rir_plan],round(query*SAMPLE_RATE))
        blended_foa=foa_direction(blend,listener,position,f.copy())['check']
        error=angular_distance(pixel_ray,blended_foa['foa_world_direction'])
        marker_geometry=angular_distance(pixel_ray,position-listener)
        image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
        draw.ellipse((centroid[0]-10,centroid[1]-10,centroid[0]+10,centroid[1]+10),outline='lime',width=3)
        v=np.array(blended_foa['foa_world_direction']);z=v@f
        projection=[640+focal*(v@r)/z,360-focal*(v@up)/z]
        draw.line((projection[0]-14,projection[1],projection[0]+14,projection[1]),fill='orange',width=3)
        draw.line((projection[0],projection[1]-14,projection[0],projection[1]+14),fill='orange',width=3)
        image_path=output/f'window_{len(reports)+1}.png';image.save(image_path)
        reports.append({'scene':scene_id,'window_index':window_index,'absolute_query_time_s':query,'local_frame':local_frame,
                        'decoded_video':str(folder/'video.mp4'),'marker_pixels':len(ids),'marker_centroid_px':centroid.tolist(),
                        'geometry_projected_px':expected.tolist(),'marker_geometry_error_deg':marker_geometry,
                        'native_foa_geometry_error_deg':native_foa['angle_deg'],'interpolated_foa_geometry_error_deg':blended_foa['angle_deg'],
                        'encoded_marker_vs_foa_error_deg':error,'pass_5deg':error <= 5,'image_path':str(image_path),
                        'method':'v6 raw W*[X,Y,Z] direct 0.5 ms; world_to_camera[forward;-right;up]; actual CPU encoded pixel centroid',
                        'formal_Habitat_RGB_verified':False})
        del renderer
    write_json(output/'checks.json',reports)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("CUDA_VISIBLE_DEVICES='' is required for this CPU-only tool")
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('--inputs',type=Path,required=True);run.add_argument('--output',type=Path,required=True)
    run.add_argument('--workers',type=int,default=2);run.add_argument('--limit-scenes',type=int,default=0)
    bench=sub.add_parser('benchmark');bench.add_argument('--inputs',type=Path,required=True);bench.add_argument('--output',type=Path,required=True)
    bench.add_argument('--workers',type=int,nargs='+',default=[1,8,32,64]);bench.add_argument('--rirs-per-worker',type=int,default=4)
    review=sub.add_parser('review-windows');review.add_argument('--inputs',type=Path,required=True);review.add_argument('--pilot-root',type=Path,required=True);review.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='benchmark':
        benchmark(args)
    elif args.command=='review-windows':
        review_windows(args)
    else:
        if not 1 <= args.workers <= 64:
            raise ValueError('maximum64 CPU workers')
        inputs=json.loads(args.inputs.read_text());args.output.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter()
        tasks=[(str(args.inputs),str(args.output),h['house'],args.limit_scenes) for h in inputs['houses']]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(house_worker,tasks))
        write_json(args.output/f'controller_receipt_{time.time_ns()}.json',{'workers':min(args.workers,len(tasks)),'wall_seconds':time.perf_counter()-start,
                   'houses':results,'timestamp':datetime.now(timezone.utc).isoformat(),'compute_device':'CPU'})


if __name__=='__main__':
    main()
