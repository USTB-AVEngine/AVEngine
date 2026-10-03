#!/usr/bin/env python3
"""CPU native paired visibility check with unchanged published house pixels."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from copy import deepcopy
import numpy as np
from avengine.timeline.continuous_pilot import make_scene,write_json
from avengine.capture.continuous_native_video import NativeContinuousVideo
from avengine.capture.continuous_native_visibility import NativeVisibilityStream

def read(path):return json.loads(Path(path).read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--encode-full',action='store_true');p.add_argument('--collect-full-visibility',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    inputs=read(a.inputs);assets=read(a.assets)['assets'];a.output.mkdir(parents=True,exist_ok=True);results=[]
    for house in inputs['houses']:
        root=a.output/house['house'];root.mkdir();scene=make_scene(house,0,30.)
        scene['sources'].pop('clap1');scene['events']=[e for e in scene['events'] if e['endpoint_id']!='clap1']
        import inspect
        kwargs={'source_material_cache_directory':root/'source_material_cache'} if 'source_material_cache_directory' in inspect.signature(NativeContinuousVideo).parameters else {}
        primary=NativeContinuousVideo(house,scene,inputs,assets,**kwargs);visibility=None
        try:
            visibility=NativeVisibilityStream(primary,root/"debug_masks")
            checks=[]
            for frame in (0,45,90):
                actors=primary._apply(scene,frame);before=primary.sim.get_sensor_observations()
                rgb=np.asarray(before['rgb']).copy();semantic=np.asarray(before['semantic']).copy()
                primary._check_source_frame(rgb[:,:,:3],semantic,frame)
                result=visibility.capture(frame,semantic,actors)
                after=primary.sim.get_sensor_observations()
                if not np.array_equal(rgb,after['rgb']) or not np.array_equal(semantic,after['semantic']):
                    raise ValueError('auxiliary visibility pass changed published house pixels')
                from PIL import Image
                Image.fromarray(rgb[:,:,:3]).save(root/f'native_frame_{frame:04d}.png')
                checks.append({'frame_index':frame,'unchanged_RGB_and_modal_pixels':True,'truth':result})
            sampled_alignment=visibility.receipt();full=None
            if a.encode_full:
                visibility.close();visibility=None
                full=primary.encode(scene,root/'scene_video.mp4',collect_pixel_visibility=a.collect_full_visibility)
            row={'full_video':{'frame_count':full['frame_count'],'source_material_checks':full['source_material_checks'],'pixel_visibility_status':full['pixel_visibility_truth']['status'],'alignment':full['native_visibility_alignment'],'wall_seconds':full['wall_seconds']} if full else None,
                'house':house['house'],'status':'pass_sampled_frames_only_full_stream_not_admitted','checks':checks,'alignment':sampled_alignment,
                'formal_dataset_count':0,'producer_only_no_QA_logic_changed':True,'source_material_checks':primary.material_checks}
            write_json(root/'summary.json',row);results.append(row)
        finally:
            if visibility:visibility.close()
            primary.close()
    write_json(a.output/'summary.json',{'houses':results,'formal_dataset_count':0})
if __name__=='__main__':main()
