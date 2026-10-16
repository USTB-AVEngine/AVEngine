"""Read UE scene captures and fail closed on occlusion, scale and placement."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from v7_human_roster_materials import json_write

def hdr(path):
 raw=path.read_bytes();offset=0
 def line():
  nonlocal offset
  end=raw.index(b'\n',offset);s=raw[offset:end];offset=end+1;return s
 magic=line()
 if not magic.startswith(b'#?'):raise ValueError('Expected Radiance HDR')
 while line():pass
 resolution=line().decode().split();assert resolution[0]=='-Y' and resolution[2]=='+X',resolution
 height,width=int(resolution[1]),int(resolution[3]);rgbe=np.empty((height,width,4),dtype=np.uint8)
 for y in range(height):
  start=raw[offset:offset+4];offset+=4
  if start[:2]!=bytes([2,2])or(start[2]<<8|start[3])!=width:
   offset-=4;rgbe[y]=np.frombuffer(raw[offset:offset+width*4],np.uint8).reshape(width,4);offset+=width*4;continue
  for c in range(4):
   x=0
   while x<width:
    n=raw[offset];offset+=1
    if n>128:
     n-=128;rgbe[y,x:x+n,c]=raw[offset];offset+=1
    else:rgbe[y,x:x+n,c]=np.frombuffer(raw[offset:offset+n],np.uint8);offset+=n
    x+=n
 exponent=rgbe[:,:,3].astype(np.int32);scale=np.ldexp(np.ones((height,width)),exponent-136);scale[exponent==0]=0
 return rgbe[:,:,:3].astype(float)*scale[:,:,None]

def project(rec,point):
 eye=np.array(rec['camera_position_cm']);target=np.array(rec['camera_target_cm']);forward=target-eye;forward/=np.linalg.norm(forward);right=np.cross([0,0,1],forward);right/=np.linalg.norm(right);up=np.cross(forward,right);v=np.array(point)-eye;z=v@forward;f=1/math.tan(math.radians(rec['fov_degrees'])/2)
 return [float((.5+(v@right)*f/z/2)*640),float((.5-(v@up)*f/z/2)*640)]

def run(T,filename):
 p=T/'evidence'/filename;rows=json.loads(p.read_text());out=[]
 for r in rows:
  dest=T/r['capture_directory']if r.get('capture_directory')else T/'renders/ue'/r['asset_id']/r['color'];r['capture_qa_status']='fail'
  try:
   rgb=np.asarray(Image.open(dest/'rgb.png').convert('RGB'));full=hdr(dest/'depth.hdr')[:,:,0];isolated=hdr(dest/'isolated_depth.hdr')[:,:,0];mask=np.isfinite(isolated)&(isolated>50)&(isolated<1000);visible=mask&(np.abs(full-isolated)<2.1);pixels=int(visible.sum());total=int(mask.sum())
   Image.fromarray(visible.astype('uint8')*255).save(dest/'visible_mask.png')
   Image.fromarray(mask.astype('uint8')*255).save(dest/'human_mask.png');mouth_pixel=project(r,r['mouth_world_cm']);marked=Image.fromarray(rgb);d=ImageDraw.Draw(marked);x,y=mouth_pixel;d.ellipse((x-4,y-4,x+4,y+4),outline=(0,255,0),width=2);marked.save(dest/'mouth.png')
   r.update(visible_pixels=pixels,isolated_human_pixels=total,occluded_pixels=total-pixels,visible_fraction=pixels/total if total else 0,mouth_pixel=mouth_pixel,highlight_ratio_rgb_ge_245=float(np.mean(np.all(rgb[visible]>=245,axis=1)))if pixels else None,visibility_method='full and human-only native linear scene depth, cm; equality tolerance 2.1cm (Radiance RGBE quantisation)',capture_qa_status='pass'if pixels>1500 and total and pixels/total>.9 else'fail')
   if r['capture_qa_status']!='pass':r['status']='fail'
  except Exception as e:r['capture_error']=repr(e);r['status']='fail'
  json_write(dest/'native.json',r);out.append(r);print('UE_CAPTURE_QA',r['asset_id'],r['color'],r['status'],r.get('visible_pixels'),r.get('visible_fraction'),flush=True)
 json_write(p,out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--result-name',default='ue_native_results.json');a=p.parse_args();run(a.task_root,a.result_name)
