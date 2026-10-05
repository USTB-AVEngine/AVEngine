#!/usr/bin/env python3
"""Measure actual full-tail CPU convolution using retained native RIR inputs."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np
from avengine.acoustics.moving_rir_calibration import correlated_convolve,effective_emission_filter,CorrelatedRIRTimeline
from avengine.acoustics.house_acceptance_metrics import foa_direction
from avengine.timeline.continuous_scene import plan_rir_keyframes,position_at,raw_world_to_camera_sn3d
from avengine.timeline.continuous_pilot import write_json
from avengine.spatial_audio.audio import write_float32_wav
from tools.acoustics.compare_moving_energy_policies import ReadOnlyRIRs


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--round3',type=Path,required=True)
    parser.add_argument('--round4',type=Path,required=True)
    parser.add_argument('--qualified_visual',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    args.output.mkdir(parents=True,exist_ok=False);results=[];directions=[]
    for house_index,house in enumerate(read(args.inputs)['houses']):
        hid=house['house'];root=args.output/hid;root.mkdir()
        scene=read(args.round3/'native_windows_correct_clock'/(hid+'__observed_scene.json'))
        track=scene['sources']['voice2']['trajectory'];listener=np.asarray(scene['listener_m']);forward=np.asarray(scene['camera_forward'])
        keys=plan_rir_keyframes(track,scene['duration_s'],listener,max_seconds=.25,max_distance_m=.5)
        cache=ReadOnlyRIRs(args.round4/'energy_four_arms'/hid/'diffraction_on_production_cache')
        filters=[cache.get(k['position_m']) for k in keys];samples=[k['sample'] for k in keys]
        length=round(scene['duration_s']*16000)
        # Procedural full-occupancy workload is a cost probe, never an admitted
        # sound-class example or a formal question.
        dry=np.random.default_rng(2026100406+house_index).normal(0,.02,length)
        start,cpu=time.perf_counter(),time.process_time()
        wet=correlated_convolve(dry,filters,samples)
        native_convolution_cpu=time.process_time()-cpu;native_convolution_wall=time.perf_counter()-start
        camera=raw_world_to_camera_sn3d(wet,forward)
        write_float32_wav(root/'scene_foa_full_tail.wav',camera,16000,channel_axis=0)
        for i,start_s in enumerate((4.,14.)):
            lo=round(start_s*16000);write_float32_wav(root/f'window_{i}_audio_foa.wav',camera[:,lo:lo+160000],16000,channel_axis=0)
        visual=read(args.qualified_visual/hid/'native_visual_evidence.json')
        visual_track=visual['observed_mouth_trajectories']['voice2']
        time_probes=(1.3666666666666667,3.3666666666666667,5.366666666666666,8.366666666666667,9.366666666666667)
        for t in time_probes[:3 if house_index==0 else 2]:
            sample=round(t*16000);pulse=np.zeros(length);pulse[sample]=1.
            actual=correlated_convolve(pulse,filters,samples)
            kernel=effective_emission_filter(CorrelatedRIRTimeline(filters,samples),sample,length)
            error=float(np.max(np.abs(actual[:,sample:sample+kernel.shape[1]]-kernel)))
            if error>1e-10:raise ValueError('effective filter differs from actual source-time convolution')
            frame_index=round(t*15);frame_t=frame_index/15
            visible_mouth=position_at(visual_track,frame_t)
            c=foa_direction(actual[:,sample:],listener,visible_mouth,forward.copy())['check']
            if c['angle_deg']>5:raise ValueError('actual FOA/observed native mouth direction fails 5deg')
            directions.append({'house':hid,'audio_emission_s':t,'video_frame_index':frame_index,
                'video_frame_time_s':frame_t,'clock_difference_s':abs(t-frame_t),
                'actual_convolution_matches_effective_filter_max_abs':error,'v6_raw_W_xyz_check':c,
                'native_visual_evidence':str(args.qualified_visual/hid/'native_visual_evidence.json')})
        receipt={'house':hid,'duration_s':scene['duration_s'],'dense_dry_samples':length,
            'native_RIR_keys':len(keys),'distinct_native_positions':len({tuple(k['position_m']) for k in keys}),
            'new_native_RIR_calls':0,'retained_native_RIRs_reused':True,
            'convolution_CPU_seconds':native_convolution_cpu,'convolution_wall_seconds':native_convolution_wall,
            'full_tail_samples':wet.shape[1],'tail_dropped':False,
            'virtual_kernel_step_samples':160,'virtual_kernel_step_s':.01,
            'formal_questions':0,'workload':'procedural dense noise, benchmark only'}
        write_json(root/'receipt.json',receipt);results.append(receipt);print(json.dumps(receipt),flush=True)
    write_json(args.output/'summary.json',{'scenes':results,'native_mouth_FOA_checks':directions,
        'formal_questions':0,'new_native_RIR_calls':0})


if __name__=='__main__':main()
