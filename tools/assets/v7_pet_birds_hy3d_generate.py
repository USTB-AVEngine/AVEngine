"""Local-weight Hunyuan3D 2.1 shape/PBR generation into one task-owned asset directory."""
import argparse,importlib.metadata,json,os,sys,time,traceback
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--task-root",type=Path,required=True);p.add_argument("--runtime-root",type=Path,required=True);p.add_argument("--weights",type=Path,required=True);p.add_argument("--kind",choices=["canary","budgerigar"],required=True);p.add_argument("--stage",choices=["shape","paint"],required=True);p.add_argument("--seed",type=int,default=20261012);p.add_argument("--octree-resolution",type=int,default=384);p.add_argument("--paint-resolution",type=int,default=512);p.add_argument("--texture-size",type=int,default=2048);p.add_argument("--disable-super-resolution",action="store_true");a=p.parse_args()
    sys.path[:0]=[str(a.task_root/"runtime/python"),str(a.runtime_root/"hy3dshape"),str(a.runtime_root/"hy3dpaint"),str(a.runtime_root)]
    os.chdir(a.runtime_root);os.environ["HF_HUB_OFFLINE"]="1";os.environ["HF_HOME"]=str(a.task_root/"cache/hf")
    import torch
    from PIL import Image
    d=a.task_root/"assets"/a.kind/"generation";d.mkdir(parents=True,exist_ok=True)
    reference=a.task_root/"assets"/a.kind/"reference/input_rgba_isnet.png"
    record={"kind":a.kind,"stage":a.stage,"model":"Tencent-Hunyuan/Hunyuan3D-2.1","model_weights":str(a.weights),"runtime_root":str(a.runtime_root),"seed":a.seed,"input_rgba":str(reference),"gpu_visible":os.environ.get("CUDA_VISIBLE_DEVICES"),"offload":False,"parameters":vars(a).copy(),"status":"running"}
    record["parameters"]={k:str(v) if isinstance(v,Path) else v for k,v in record["parameters"].items()}
    t0=time.time();torch.cuda.reset_peak_memory_stats()
    try:
        if a.stage=="shape":
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dshape.postprocessors import DegenerateFaceRemover,FaceReducer
            pipe=Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(str(a.weights),device="cuda",dtype=torch.float16)
            record["pipeline_constructed"]=True
            mesh=pipe(image=Image.open(reference).convert("RGBA"),num_inference_steps=50,octree_resolution=a.octree_resolution,num_chunks=8000,generator=torch.Generator("cuda").manual_seed(a.seed))[0]
            mesh.export(d/f"shape_raw_seed{a.seed}.glb")
            record.update(raw_vertices=len(mesh.vertices),raw_triangles=len(mesh.faces),raw_bounds=mesh.bounds.tolist())
            mesh=DegenerateFaceRemover()(mesh);mesh=FaceReducer()(mesh,max_facenum=40000);mesh.export(d/"shape.glb")
            record.update(vertices=len(mesh.vertices),triangles=len(mesh.faces),bounds=mesh.bounds.tolist(),output=str(d/"shape.glb"))
        else:
            from textureGenPipeline import Hunyuan3DPaintPipeline,Hunyuan3DPaintConfig
            import textureGenPipeline
            conf=Hunyuan3DPaintConfig(6,a.paint_resolution);conf.seed=a.seed;conf.multiview_pretrained_path=str(a.weights);conf.dino_ckpt_path=str(a.weights/"dependencies/dinov2-giant");conf.realesrgan_ckpt_path=str(a.weights/"dependencies/realesrgan/RealESRGAN_x4plus.pth")
            conf.multiview_cfg_path=str(a.runtime_root/"hy3dpaint/cfgs/hunyuan-paint-pbr.yaml");conf.custom_pipeline=str(a.runtime_root/"hy3dpaint/hunyuanpaintpbr");conf.render_size=1024;conf.texture_size=a.texture_size
            if a.disable_super_resolution:
                class IdentitySuper:
                    def __init__(self,config):pass
                    def __call__(self,image):return image
                textureGenPipeline.imageSuperNet=IdentitySuper
            torch.manual_seed(a.seed);pipe=Hunyuan3DPaintPipeline(conf);record["pipeline_constructed"]=True
            out=pipe(mesh_path=str(d/"shape.glb"),image_path=str(reference),output_mesh_path=str(d/"textured.obj"),use_remesh=False,save_glb=False)
            record["output"]=str(out);record["sidecars"]=[str(x) for x in d.glob("textured*")]
        record["status"]="pass"
    except Exception:
        record.update(status="fail",error=traceback.format_exc());raise
    finally:
        record.update(wall_seconds=time.time()-t0,peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576,packages={x:importlib.metadata.version(x) for x in ["torch","diffusers","transformers","trimesh"]})
        (d/f"{a.stage}_seed{a.seed}.json").write_text(json.dumps(record,indent=2)+"\n");print(json.dumps(record),flush=True)
if __name__=="__main__":main()
