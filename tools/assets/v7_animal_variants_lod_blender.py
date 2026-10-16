"""CPU-only static animal LOD; preserve generated shape and texture coordinates."""
import argparse
import json
import sys
from pathlib import Path
import bpy

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(a.input))
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    before=sum(sum(len(f.vertices)-2 for f in o.data.polygons) for o in meshes)
    ratio=min(1.,85000/max(before,1))
    for obj in meshes:
        bpy.context.view_layer.objects.active=obj;obj.select_set(True)
        if ratio<1:
            mod=obj.modifiers.new('OwnedTaskStaticLOD','DECIMATE');mod.ratio=ratio;mod.use_collapse_triangulate=True;bpy.ops.object.modifier_apply(modifier=mod.name)
        obj.select_set(False)
    textures=[]
    for image in bpy.data.images:
        if image.type!='IMAGE' or image.size[0]==0:continue
        source_size=list(image.size)
        if max(image.size)>2048:
            scale=2048/max(image.size);image.scale(round(image.size[0]*scale),round(image.size[1]*scale))
        image.file_format='PNG';textures.append({'name':image.name,'original_size':source_size,'lod_size':list(image.size)})
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type=='BSDF_PRINCIPLED':
                    for key,val in [('Metallic',0.),('Roughness',1.)]:
                        socket=node.inputs.get(key)
                        for link in list(socket.links):material.node_tree.links.remove(link)
                        socket.default_value=val
    a.output.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(a.output),export_format='GLB',export_image_format='AUTO',export_animations=False,export_apply=True)
    after=sum(sum(len(f.vertices)-2 for f in o.data.polygons) for o in meshes)
    a.receipt.write_text(json.dumps({'blender_version':bpy.app.version_string,'input':str(a.input),'output':str(a.output),'input_triangles':before,'lod_triangles':after,'decimation_ratio':ratio,'textures':textures,'method':'CPU quadric collapse with UVs; generated animal retained; no donor silhouette'},indent=2)+'\n')
    if after>100000:raise RuntimeError('LOD still exceeds triangle budget')
if __name__=='__main__':main()
