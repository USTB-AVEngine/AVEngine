"""CPU reuse of existing orthographic RGB frames; missing pixels remain unverified."""
from pathlib import Path
import json,math
import numpy as np
import shapely
from shapely.geometry import Polygon,shape
from PIL import Image


def frame_polygon(entry,im,floor_y):
    clip=np.asarray(im['projection'])@np.asarray(entry['view'])
    if not np.allclose(clip[3],[0,0,0,1]):raise ValueError('Only orthographic cached RGB can be reused')
    A=clip[:2][:,[0,2]];offset=clip[:2,1]*floor_y+clip[:2,3]
    return Polygon([np.linalg.solve(A,np.array(q)-offset) for q in ((-1,-1),(1,-1),(1,1),(-1,1))])


def floor_overhead(row,floor,cache,output,base=None):
    """Use an original frame when complete, otherwise a world-coordinate mosaic.

    No synthetic RGB is used to estimate missing scan evidence. A mosaic made
    from existing real orthographic views is resampling, not a new rendering.
    """
    floor_y=floor['floor_y_m'];scope=shape(floor['floor_polygon']);cache=Path(cache);output=Path(output)
    rendered=output.parent/'overhead_cpu_render_v1';render_key=row['house']+'__'+row['room_label']+'__'+floor['floor_id'];render_meta=rendered/(render_key+'.json');render_png=rendered/(render_key+'.png')
    if render_meta.exists() and render_png.exists():
        entry=json.loads(render_meta.read_text());im=entry['images'][0];image=Image.open(render_png).convert('RGB')
        return entry,im,image,{'metadata_path':str(render_meta),'image_path':str(render_png),'source':entry['source'],'orientation':entry['orientation'],'gl_renderer':entry['gl_renderer'],'compute_device':'CPU'}
    if base is not None and abs(base[0]['floor_y_m']-floor_y)<=0.3:
        frame=frame_polygon(base[0],base[1],floor_y)
        if scope.difference(frame).area<=max(1e-8,scope.area*1e-6):return base
    key=row['house']+'__'+row['room_label']+'__'+floor['floor_id'];meta=output/(key+'.json');png=output/(key+'.png')
    if meta.exists():
        entry=json.loads(meta.read_text())
        if not png.exists():return None
        im=entry['images'][0];image=Image.open(png).convert('RGB')
        return entry,im,image,{'metadata_path':str(meta),'image_path':str(png),'source':'CPU orthographic mosaic of existing readonly frames','orientation':entry['orientation'],'source_frames':entry['source_frames'],'world_floor_in_frame_fraction':entry['world_floor_in_frame_fraction']}
    frames=[]
    for path in cache.glob(row['house']+'__*.json'):
        entry=json.loads(path.read_text());source_png=path.with_suffix('.png')
        if abs(entry['floor_y_m']-floor_y)>0.3 or not source_png.exists():continue
        im=next((x for x in entry['images'] if x['path'].endswith('_overview.png')),None)
        if im is None:continue
        try:frame=frame_polygon(entry,im,floor_y)
        except (ValueError,np.linalg.LinAlgError):continue
        if frame.intersection(scope).area<=1e-8:continue
        frames.append((entry,im,source_png,path,frame))
    if not frames:return None
    covered=shapely.union_all([f[4] for f in frames]);fraction=float(scope.intersection(covered).area/scope.area) if scope.area else 0
    x0,z0,x1,z1=scope.bounds;cx=(x0+x1)/2;cz=(z0+z1)/2;span=max(x1-x0,z1-z0,2)*1.10
    size=1024;xx,yy=np.meshgrid(np.arange(size)+0.5,np.arange(size)+0.5)
    wx=cx+(0.5-xx/size)*span;wz=cz+(0.5-yy/size)*span
    rgb=np.zeros((size,size,3),np.uint8);assigned=np.zeros((size,size),bool)
    # Prefer the closest source floor height, then the highest pixel density.
    frames.sort(key=lambda f:(abs(f[0]['floor_y_m']-floor_y),-1024/f[1]['span_m'],str(f[3])))
    sources=[]
    for entry,im,path,metadata,frame in frames:
        clip=np.asarray(im['projection'])@np.asarray(entry['view'])
        nx=clip[0,0]*wx+clip[0,2]*wz+clip[0,1]*floor_y+clip[0,3]
        ny=clip[1,0]*wx+clip[1,2]*wz+clip[1,1]*floor_y+clip[1,3]
        image=np.asarray(Image.open(path).convert('RGB'));height,width=image.shape[:2]
        px=(nx+1)*width/2;py=(1-ny)*height/2
        good=(px>=0)&(px<width)&(py>=0)&(py<height)&~assigned
        ix=np.clip(np.floor(px).astype(int),0,width-1);iy=np.clip(np.floor(py).astype(int),0,height-1)
        rgb[good]=image[iy[good],ix[good]];assigned[good]=True
        sources.append({'metadata_path':str(metadata),'image_path':str(path),'floor_y_m':entry['floor_y_m'],'orientation':entry.get('orientation'),'original_span_m':im['span_m'],'assigned_pixel_count':int(good.sum())})
    entry={'house':row['house'],'source_region':row['room_label'],'floor_id':floor['floor_id'],'floor_y_m':floor_y,'orientation':'image_right=-X,image_up=+Z','size_px':[size,size],'view':[[-1,0,0,cx],[0,0,1,-cz],[0,1,0,-floor_y-30],[0,0,0,1]],'images':[{'path':str(png),'span_m':span,'projection':np.diag([2/span,2/span,1,1]).tolist(),'cut_height_above_floor_m':1.8}],'source_frames':sources,'world_floor_in_frame_fraction':fraction,'coverage_geometry_xz_m':shapely.geometry.mapping(covered),'pixel_method':'nearest-neighbour world-coordinate resampling of original orthographic RGB; no invented geometry or texture','compute_device':'CPU','software_rendered_new_scene':False,'habitat_new_rendering':'not_run; real cached frames reused'}
    output.mkdir(parents=True,exist_ok=True)
    # Each source/floor is owned by one house worker; artifacts are never replaced.
    with png.open('xb') as f:Image.fromarray(rgb).save(f,format='PNG')
    with meta.open('x') as f:json.dump(entry,f,ensure_ascii=False)
    return entry,entry['images'][0],Image.fromarray(rgb),{'metadata_path':str(meta),'image_path':str(png),'source':'CPU orthographic mosaic of existing readonly frames','orientation':entry['orientation'],'source_frames':sources,'world_floor_in_frame_fraction':fraction}
