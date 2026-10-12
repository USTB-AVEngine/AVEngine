"""Read-only scan semantics certify attachment types against original geometry."""
from collections import defaultdict
from pathlib import Path
import csv,json,struct
import numpy as np


def category_kind(name,canonical=None):
    import re
    name=re.sub(r'[_#]+',' ',name.strip().lower())
    # MP3D raw names preserve furniture detail that its coarser taxonomy can
    # map to chest_of_drawers (e.g. a bedside table). Both are original labels.
    if canonical in {'table','counter','desk'}:return 'table'
    if name in {'table','desk','counter','countertop','nighttable','endtable','sidetable'} or re.fullmatch(r'(?:coffee|dining|kitchen|end|side|bedside|night|office|computer|small|round|wall|sink|cabinet|folding|ping pong|coffe|dinner) (?:table|desk|counter|countertop)',name):return 'table'
    # Ground coverings remain floor only after the independent native measured
    # floor-level and free-floor polygon checks. Furniture is never an alias.
    if name in {'carpet','rug','doormat','floor mat','floor rug','floor carpet'}:return 'floor'
    if name in {'floor','ceiling','wall'}:return name
    if 'door' in name:return 'door'
    if 'window' in name:return 'window'
    return None


def _accessor(d,b,index):
    a=d['accessors'][index];v=d['bufferViews'][a['bufferView']];dt=np.dtype({5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]);width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
    return np.ndarray((a['count'],width),dtype=dt,buffer=b,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',dt.itemsize*width),dt.itemsize))


class SemanticSupports:
    def __init__(self,house):
        import trimesh
        self.meshes={};self.anchors=defaultdict(list);self.sources=[];chunks=defaultdict(list);vertices=defaultdict(list);base=Path(house['input']['scene_directory']);scan=house['input']['scan_id']
        if house['family']=='hm3d':
            label=base/(scan+'.semantic.txt');path=base/(scan+'.semantic.glb');self.sources=[str(label),str(path)]
            rows=list(csv.reader(label.open()))[1:];lookup={tuple(bytes.fromhex(r[1])):(r[0],category_kind(r[2]),r[2]) for r in rows}
            with path.open('rb') as f:
                f.read(12);n,t=struct.unpack('<II',f.read(8));d=json.loads(f.read(n));n,t=struct.unpack('<II',f.read(8));b=f.read(n)
            if any(set(node)-{'name','mesh','children'} for node in d['nodes']):raise ValueError('semantic scan has an unhandled native transform')
            for mesh in d['meshes']:
                for p in mesh['primitives']:
                    raw_points=_accessor(d,b,p['attributes']['POSITION']).astype(float);points=np.column_stack([raw_points[:,0],raw_points[:,2],-raw_points[:,1]]);tri=_accessor(d,b,p['indices']).reshape(-1,3);colors=_accessor(d,b,p['attributes']['COLOR_0'])[:,:3]/65535
                    rgb=np.rint(np.where(colors<=.0031308,colors*12.92,1.055*colors**(1/2.4)-.055)*255).astype(np.uint8)
                    packed=rgb[:,0].astype(np.uint32)*65536+rgb[:,1].astype(np.uint32)*256+rgb[:,2];same=(packed[tri[:,0]]==packed[tri[:,1]])&(packed[tri[:,1]]==packed[tri[:,2]])
                    for key in np.unique(packed):
                        col=((int(key)>>16)&255,(int(key)>>8)&255,int(key)&255);row=lookup.get(col)
                        if not row or not row[1]:continue
                        oid,kind,name=row;selected=tri[same&(packed[tri[:,0]]==key)]
                        if len(selected):chunks[kind].append(points[selected]);vertices[(oid,kind,name)].append(points[packed==key])
        elif house['family']=='mp3d':
            from plyfile import PlyData
            label=base/(scan+'.house');path=base/(scan+'_semantic.ply');self.sources=[str(label),str(path)];categories={};objects={}
            lines=label.read_text().splitlines()
            for line in lines:
                row=line.split()
                if row and row[0]=='C':categories[int(row[1])]=(category_kind(row[3],row[5]),row[3])
            for line in lines:
                row=line.split()
                if row and row[0]=='O':objects[int(row[1])]=categories.get(int(row[3]),(None,'unknown'))
            data=PlyData.read(path);v=data['vertex'].data;points=np.column_stack([v['x'],v['z'],-v['y']]);faces=data['face'].data
            for oid in np.unique(faces['object_id']):
                kind,name=objects.get(int(oid),(None,'unknown'))
                if kind is None:continue
                ids=np.vstack(faces['vertex_indices'][faces['object_id']==oid]);chunks[kind].append(points[ids]);vertices[(str(oid),kind,name)].append(points[np.unique(ids)])
        else:raise ValueError('only admitted HM3D/MP3D scan semantics are supported')
        for kind,parts in chunks.items():
            tri=np.concatenate(parts);self.meshes[kind]=trimesh.Trimesh(tri.reshape(-1,3),np.arange(len(tri)*3).reshape(-1,3),process=False)
        for (oid,kind,name),parts in vertices.items():
            if kind not in {'door','window'}:continue
            points=np.concatenate(parts);lo=points.min(0);hi=points.max(0)
            if kind=='door' and not 1.45<=hi[1]-lo[1]<=3.:continue
            self.anchors[kind].append({'kind':kind,'authority':'original scan semantic-labelled triangles','native_object_id':oid,'category':name,'center_m':((lo+hi)/2).tolist(),'sizes_m':(hi-lo).tolist(),'source_files':self.sources})

    def qualify_points(self,kind,points):
        import trimesh
        mesh=self.meshes.get(kind)
        if mesh is None or not points:return []
        q=np.asarray([p['point'] for p in points]);_,distance,_=trimesh.proximity.closest_point(mesh,q)
        return [{**p,'semantic_support':{'status':'pass','surface':kind,'nearest_labelled_triangle_distance_m':float(e),'threshold_m':.015,'sources':self.sources}} for p,e in zip(points,distance) if e<=.015]
