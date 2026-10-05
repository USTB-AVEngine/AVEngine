#!/usr/bin/env python3
"""Read five encoded native windows and compare head pixels to native FOA."""
from __future__ import annotations
import argparse,json,math,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from avengine.capture.continuous_native_video import NativeContinuousVideo
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.continuous_scene import camera_basis,position_at,angular_distance
from avengine.acoustics.house_acceptance_metrics import foa_direction
from tools.timeline.pilot_continuous_scenes import native_renderer

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--native-root',type=Path,required=True)
    p.add_argument('--windows-root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    inputs=json.loads(a.inputs.read_text());assets=json.loads(a.assets.read_text())['assets'];rows=[];images=[]
    selections=[(0,0,72),(0,1,222),(0,2,366),(1,0,81),(1,1,219)]
    for hi,house in enumerate(inputs['houses']):
        case=a.native_root/house['house'];scene=json.loads((case/'scene_plan_input.json').read_text())
        evidence=json.loads((case/'native_visual_evidence.json').read_text())
        video=NativeContinuousVideo(house,scene,inputs,assets)
        renderer,identity=native_renderer(inputs,house)
        for _,window_index,index in [s for s in selections if s[0]==hi]:
            start=round(scene['window_starts_s'][window_index]*5)/5;wi=index-round(start*15)
            folder=a.windows_root/(scene['scene_id']+f'__w{window_index:02d}')
            data=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-threads','1','-i',str(folder/'video.mp4'),
                '-vf',f'select=eq(n\\,{wi})','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'])
            decoded=np.frombuffer(data,dtype=np.uint8).reshape(720,1280,3)
            applied=video._apply(scene,index);obs=video.sim.get_sensor_observations()
            raw=np.asarray(obs['rgb'])[:,:,:3];semantic=np.asarray(obs['semantic'])
            target=next(x for x in applied if x['endpoint_id']=='voice2')
            original=next(x for x in evidence['frames'][index]['actors'] if x['endpoint_id']=='voice2')
            if np.max(np.abs(np.array(target['emitter_position_m'])-original['emitter_position_m']))>1e-6:
                raise ValueError('native review did not replay the captured emitter pose')
            mask=semantic==target['semantic_id'];ys,xs=np.nonzero(mask)
            if len(xs)<100:raise ValueError('native target lacks visible actor pixels')
            # Independent semantic head pixels, not the planned analytic marker.
            head=mask.copy();head[round(ys.min()+.18*(ys.max()-ys.min()+1)):]=False
            hy,hx=np.nonzero(head)
            if len(hx)<12:raise ValueError('native head pixel region is too small')
            cx,cy=float(hx.mean()),float(hy.mean());focal=1280/(2*math.tan(math.radians(85/2)))
            pixel_left_deg=-math.degrees(math.atan2(cx-640,focal))
            point=np.asarray(original['emitter_position_m']);listener=np.asarray(scene['listener_m'])
            batch=renderer.render([point.tolist()]);check=foa_direction(batch.samples[0],listener,point,np.asarray(scene['camera_forward']))['check']
            foa_left=check['foa_azimuth_left_positive_deg']
            error=abs((pixel_left_deg-foa_left+180)%360-180)
            forward,right,up=camera_basis(scene['camera_forward'])
            pixel_ray=forward+(cx-640)/focal*right-(cy-360)/focal*up
            error_3d=angular_distance(pixel_ray,np.asarray(check['foa_world_direction']))
            mae=float(np.abs(decoded.astype(float)-raw.astype(float)).mean())
            # In a CRF18 encoded frame, foreground texture/pose must agree with
            # an independently replayed native capture, not a blank frame.
            match=mae<=8 and error<=5 and error_3d<=5 and check['angle_deg']<=5
            speech_active=any(e['endpoint_id']=='voice2' and e['start_tick']<=index/15*10<e['end_tick'] for e in scene['events'])
            row={'house':house['house'],'window_index':window_index,'window_time_s':wi/15,'scene_time_s':index/15,
                 'moving_voice_active':bool(speech_active),'native_visible_pixels':len(xs),'native_head_pixels':len(hx),
                 'head_centroid_px':[cx,cy],'pixel_azimuth_left_deg':pixel_left_deg,'native_FOA_azimuth_left_deg':foa_left,
                 'pixel_vs_FOA_abs_error_deg':error,'pixel_vs_FOA_3d_error_deg':error_3d,'native_FOA_geometry_error_deg':check['angle_deg'],
                 'encoded_vs_native_rgb_MAE':mae,'status':'pass' if match and speech_active else 'fail'}
            rows.append(row)
            canvas=Image.fromarray(decoded);draw=ImageDraw.Draw(canvas)
            draw.ellipse((cx-12,cy-12,cx+12,cy+12),outline='red',width=3)
            draw.text((15,15),f'{house["house"]} w{window_index} error={error:.3f} deg',fill='red')
            canvas.save(a.output/f'window_{len(rows)}.png');images.append(canvas.resize((640,360)))
        video.close();del renderer
    montage=Image.new('RGB',(1280,3*360),'white')
    for i,img in enumerate(images):montage.paste(img,((i%2)*640,(i//2)*360))
    montage.save(a.output/'rgb_foa_review.png')
    result={'status':'pass' if all(x['status']=='pass' for x in rows) else 'fail','windows':rows,
        'max_pixel_vs_FOA_abs_error_deg':max(x['pixel_vs_FOA_abs_error_deg'] for x in rows),
        'max_pixel_vs_FOA_3d_error_deg':max(x['pixel_vs_FOA_3d_error_deg'] for x in rows),
        'native_runtime_no_gpu':True,'formal_full_scene_motion_and_asset_admission':'separate_required'}
    write_json(a.output/'summary.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
