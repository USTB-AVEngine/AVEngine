"""CPU orthographic texture rasterization and native floor-reference probes.

Black pixels retain the frozen RGB<8 definition; missing texture evidence is
unknown rather than a neutral-color pass. No Simulator or GPU is constructed.
"""
from pathlib import Path
import os, struct, subprocess
import numpy as np
from PIL import Image
import trimesh
from numba import njit

@njit(cache=False)
def raster(triangles,uvs,texture,solid,span,cx,cz,low,high,rgb,height,known):
    size=rgb.shape[0]
    for i in range(len(triangles)):
        t=triangles[i]
        if np.min(t[:,1])>high or np.max(t[:,1])<low:continue
        ax=(t[0,0]-cx)/span*size+size/2; ay=(t[0,2]-cz)/span*size+size/2
        bx=(t[1,0]-cx)/span*size+size/2; by=(t[1,2]-cz)/span*size+size/2
        dx=(t[2,0]-cx)/span*size+size/2; dy=(t[2,2]-cz)/span*size+size/2
        denominator=(by-dy)*(ax-dx)+(dx-bx)*(ay-dy)
        if abs(denominator)<1e-12:continue
        x0=max(0,int(np.floor(min(ax,bx,dx))));x1=min(size-1,int(np.ceil(max(ax,bx,dx))))
        y0=max(0,int(np.floor(min(ay,by,dy))));y1=min(size-1,int(np.ceil(max(ay,by,dy))))
        for y in range(y0,y1+1):
            for x in range(x0,x1+1):
                a=((by-dy)*(x+.5-dx)+(dx-bx)*(y+.5-dy))/denominator
                b=((dy-ay)*(x+.5-dx)+(ax-dx)*(y+.5-dy))/denominator
                c=1-a-b
                if a<0 or b<0 or c<0:continue
                h=a*t[0,1]+b*t[1,1]+c*t[2,1]
                if h<low or h>high or h<=height[y,x]:continue
                height[y,x]=h
                if len(texture):
                    u=a*uvs[i,0,0]+b*uvs[i,1,0]+c*uvs[i,2,0]
                    v=a*uvs[i,0,1]+b*uvs[i,1,1]+c*uvs[i,2,1]
                    # GLTF repeat sampler; coordinates at UV=1 wrap to UV=0.
                    tx=min(texture.shape[1]-1,int((u-np.floor(u))*texture.shape[1]))
                    ty=min(texture.shape[0]-1,int((1-(v-np.floor(v)))*texture.shape[0]))
                    rgb[y,x]=texture[ty,tx];known[y,x]=True
                elif len(solid):rgb[y,x]=solid[i];known[y,x]=True
                else:rgb[y,x]=0;known[y,x]=False

def basis_material_images(path, decoder):
    """Decode GOOGLE_texture_basis with a declared external CPU executable.

    Material names preserve the raw GLTF identity consumed by Trimesh. Encoded
    raster rows are retained and the Basis Y-flipped flag is recorded. Native
    Habitat RGB equivalence is unverified; no room transform is inferred.
    """
    from avengine.acoustics.gltf import load_glb_bytes
    glb=load_glb_bytes(Path(path).read_bytes(),source_path=str(path));doc=glb.document
    decoded={};materials={};diagnostics=[]
    for material in doc.get('materials',[]):
        index=material.get('pbrMetallicRoughness',{}).get('baseColorTexture',{}).get('index')
        if index is None:continue
        texture=doc['textures'][index]
        source=texture.get('extensions',{}).get('GOOGLE_texture_basis',{}).get('source')
        if source is None:continue
        image=doc['images'][source]
        if image.get('mimeType')!='image/x-basis':continue
        name=material.get('name')
        if not name:raise ValueError('Basis material requires its GLTF material name')
        if source not in decoded:
            view=doc['bufferViews'][image['bufferView']];start=view.get('byteOffset',0)
            payload=glb.binary[start:start+view['byteLength']]
            cp=subprocess.run([str(decoder)],input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,check=True)
            if len(cp.stdout)<12:raise ValueError('truncated CPU Basis response')
            width,height,flipped=struct.unpack_from('<III',cp.stdout)
            if width<1 or height<1 or len(cp.stdout)!=12+width*height*4:raise ValueError('invalid CPU Basis response dimensions')
            rgb=np.frombuffer(cp.stdout,dtype=np.uint8,offset=12).reshape(height,width,4)[...,:3].copy()
            decoded[source]=Image.fromarray(rgb)
            diagnostics.append(dict(image_index=source,name=image.get('name'),size=[width,height],basis_y_flipped=bool(flipped),rgba_level=0,decoder=str(decoder),black_texel_fraction=float(np.all(rgb<8,axis=-1).mean())))
        im=decoded[source]
        if name in materials and materials[name] is not im:raise ValueError('conflicting raw Basis material name')
        materials[name]=im
    return materials,diagnostics

class CPUOverhead:
    def __init__(self,path):
        self.path=Path(path);self.items=[];self.basis_diagnostics=[];self.basis_error=None
        basis={};decoder=os.environ.get('AVENGINE_BASIS_CPU_DECODER')
        if decoder:
            try:basis,self.basis_diagnostics=basis_material_images(path,decoder)
            except Exception as exc:self.basis_error=type(exc).__name__+': '+str(exc)
        scene=trimesh.load_scene(path,process=False)
        for m in scene.dump():
            if not isinstance(m,trimesh.Trimesh):continue
            vertices=np.asarray(m.vertices,float)
            vertices=np.stack((vertices[:,0],vertices[:,2],-vertices[:,1]),axis=-1)
            triangles=vertices[np.asarray(m.faces)]
            tex=np.empty((0,0,3),np.uint8);uv=np.zeros((len(triangles),3,2));solid=np.empty((0,3),np.uint8)
            if m.visual.kind=='texture':
                material=m.visual.material;im=getattr(material,'image',None)
                if im is None:im=basis.get(getattr(material,'name',None))
                if im is None and hasattr(material,'to_simple'):im=material.to_simple().image
                if im is not None and m.visual.uv is not None:
                    tex=np.asarray(im.convert('RGB'));uv=np.asarray(m.visual.uv)[m.faces].astype(float)
            elif m.visual.kind in ('face','vertex'):solid=np.asarray(m.visual.face_colors)[:,:3].astype(np.uint8)
            self.items.append((triangles,uv,tex,solid))
        if not self.items:raise ValueError('no raw visual triangles')

    def render(self,scope,floor_y,output,p,size=512):
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        x0,z0,x1,z1=scope.bounds;cx=(x0+x1)/2;cz=(z0+z1)/2
        span=max(x1-x0,z1-z0)*1.1
        rgb=np.zeros((size,size,3),np.uint8);height=np.full((size,size),-np.inf);known=np.ones((size,size),bool)
        for t,uv,tex,solid in self.items:
            raster(t,uv,tex,solid,span,cx,cz,floor_y-p['floor_height_separation_m'],floor_y+p['camera_height_m'],rgb,height,known)
        entry={'floor_y_m':float(floor_y),'orientation':'image_right=+X,image_down=+Z',
            'view':[[1,0,0,-cx],[0,0,-1,cz],[0,1,0,-floor_y],[0,0,0,1]]}
        im={'projection':np.diag([2/span,2/span,1,1]).tolist(),'span_m':span,'path':str(output)}
        from .media import polygon_mask
        mask=polygon_mask(scope,floor_y,entry,im,(size,size))
        missing=int((mask & ~known).sum())
        Image.fromarray(rgb).save(output)
        diag={'backend':'CPU orthographic raw textured triangles; per-pixel barycentric UV, nearest texel','image_path':str(output),'size':[size,size],'slab_habitat_y_m':[float(floor_y-p['floor_height_separation_m']),float(floor_y+p['camera_height_m'])],'unavailable_texture_pixels_in_scope':missing,'covered_pixels_in_scope':int((mask & np.isfinite(height)).sum()),'compute_device':'CPU','gpu_renderer_created':False,'habitat_rgb_equivalence':'unverified','basis_textures':self.basis_diagnostics,'basis_decoder_error':self.basis_error}
        if missing:diag['black_measurement_usable']=False
        else:diag['black_measurement_usable']=True
        return entry,im,Image.fromarray(rgb),diag

def floor_reference(mesh,pf,scope,floor_y,p,native_floor_y=None):
    """Probe the existing native navmesh and raw scan before room-grid sampling."""
    import shapely
    x0,z0,x1,z1=scope.bounds
    points=[scope.representative_point()]
    from shapely.geometry import Point
    points += [Point(x,z) for x in np.linspace(x0,x1,5)[1:-1] for z in np.linspace(z0,z1,5)[1:-1] if shapely.contains_xy(scope,x,z)]
    records=[]
    for point in points:
        probe=np.array([point.x,floor_y,point.y],np.float32);q=np.asarray(pf.snap_point(probe),float)
        if not np.isfinite(q).all() or np.linalg.norm(q[[0,2]]-probe[[0,2]])>p['nav_snap_horizontal_max_m'] or abs(q[1]-floor_y)>p['floor_height_separation_m']:continue
        if not shapely.contains_xy(scope,q[0],q[2]):continue
        origin=q+[0,.3,0]
        hits,_,_=mesh.ray.intersects_location([origin],[[0,-1,0]],multiple_hits=False)
        raw_y=float(hits[0,1]) if len(hits) else None
        records.append(dict(probe_m=probe.tolist(),native_floor_m=q.tolist(),semantic_to_nav_offset_m=float(q[1]-floor_y),raw_downward_hit_y_m=raw_y,raw_to_nav_offset_m=float(raw_y-q[1]) if raw_y is not None else None))
    return dict(status='measured' if records else 'review',method='existing Habitat PathFinder snap plus first raw-GLB downward ray; no Simulator',semantic_floor_y_m=float(floor_y),native_house_floor_y_m=native_floor_y,probes=records,geometry_shift_applied_m=0.,measured_before_room_grid_and_placement=True)
