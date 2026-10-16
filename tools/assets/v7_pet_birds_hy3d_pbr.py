"""Export Hunyuan's UV mesh and generated albedo/metal/roughness maps as a PBR GLB."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
def main():
    p=argparse.ArgumentParser();p.add_argument("--generation-dir",type=Path,required=True);a=p.parse_args();d=a.generation_dir
    mesh=trimesh.load(d/"textured.obj",force="mesh",process=False)
    color=Image.open(d/"textured.jpg").convert("RGB")
    rough=Image.open(d/"textured_roughness.jpg").convert("L");metal=Image.open(d/"textured_metallic.jpg").convert("L")
    mr=np.zeros((*np.asarray(rough).shape,4),dtype=np.uint8);mr[:,:,0]=255;mr[:,:,1]=np.asarray(rough);mr[:,:,2]=np.asarray(metal);mr[:,:,3]=255
    pbr=trimesh.visual.material.PBRMaterial(name="Hunyuan3D21GeneratedPBR",baseColorTexture=color,metallicRoughnessTexture=Image.fromarray(mr),metallicFactor=1,roughnessFactor=1,doubleSided=False)
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=mesh.visual.uv,material=pbr)
    mesh.export(d/"textured_pbr.glb")
    data={"input":str(d/"textured.obj"),"output":str(d/"textured_pbr.glb"),"triangles":len(mesh.faces),"vertices":len(mesh.vertices),"bounds":mesh.bounds.tolist(),"albedo":str(d/"textured.jpg"),"generated_metallic":str(d/"textured_metallic.jpg"),"generated_roughness":str(d/"textured_roughness.jpg"),"texture_size":list(color.size),"pbr":"glTF metallic-roughness G=roughness B=metallic; retained generated albedo and UV","production_material_status":"pending_source_materials"}
    (d/"pbr_export.json").write_text(json.dumps(data,indent=2)+"\n");print(json.dumps(data))
if __name__=="__main__":main()
