"""Generate bounded, seed-recorded Stable Audio Open candidate WAVs offline."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, pathlib, time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def dump(path, value):
    pathlib.Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

def progress(root, text):
    with (root / "PROGRESS.md").open("a") as stream:
        stream.write("- " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + text + "\n")

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def candidate_plan(config, round_number, selected):
    tasks = []
    for c, spec in config["classes"].items():
        if selected and c not in selected:
            continue
        count = spec["needed"] * (3 if round_number == 1 else 1)
        active = min(12, max(1, count // 2))
        for i in range(count):
            bank = spec["retry_prompts"] if round_number == 2 else spec["prompts"]
            assert len(bank) >= 12
            p = copy.deepcopy(bank[i % active])
            tasks.append(dict(sound_asset_id=f"sao_r{round_number}_{c}_{i:04d}",
                original_id=f"sao_r{round_number}_{c}_{i:04d}", sound_class=c,
                ordinal=i, round=round_number, prompt_id=p["id"],
                prompt=p["text"], duration_s=p["duration_s"]))
    # Cycle classes so an interrupted run has coverage across the whole request.
    order = {c:i for i,c in enumerate(config["classes"])}
    return sorted(tasks, key=lambda r:(r["ordinal"], order[r["sound_class"]]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=pathlib.Path, required=True)
    p.add_argument("--model-dir", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--round", type=int, choices=[1,2], default=1)
    p.add_argument("--classes", nargs="*", default=[])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--deadline-utc", type=float, required=True)
    p.add_argument("--t5-cache", type=pathlib.Path, required=True)
    args=p.parse_args(); root=args.output
    config=json.loads(args.config.read_text())
    assert config["steps"] == 100 and config["cfg_scale"] == 7
    assert config["allowed_splits"] == ["train"]
    assert 1 <= args.batch_size <= 2
    start=time.time()
    # No model/cache/download directory is ever created under the shared model root.
    weight=args.model_dir/"model.safetensors"
    if not weight.is_file() or "DONE sha256 ok" not in (args.model_dir/"fetch.log").read_text():
        raise RuntimeError("official weight download has not completed and verified")
    digest=sha(weight)
    if digest != config["model_sha256"]:
        raise RuntimeError("official weight sha256 mismatch")
    import torch
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    import importlib.metadata
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    mc=json.loads((args.model_dir/"model_config.json").read_text())
    t5_ref=(args.t5_cache/"refs"/"main").read_text().strip()
    t5_path=args.t5_cache/"snapshots"/t5_ref
    if not t5_path.is_dir():
        raise RuntimeError("offline t5-base snapshot is missing")
    for cond in mc["model"]["conditioning"]["configs"]:
        if cond["type"] == "t5":
            cond["config"]["model_path"] = str(t5_path)
    print("MODEL_CREATE", flush=True)
    model=create_model_from_config(mc)
    model.load_state_dict(load_ckpt_state_dict(str(weight)), strict=True)
    model=model.eval().requires_grad_(False).to("cuda")
    print("MODEL_READY", round(time.time()-start,2), flush=True)
    progress(root, f"Stable Audio Open 模型与离线 T5 加载成功；第 {args.round} 轮生成开始。")
    tasks=candidate_plan(config,args.round,set(args.classes))
    dump(root/f"candidate_plan_r{args.round}.json",tasks)
    dump(root/("prompt_library.json" if args.round == 1 else "prompt_library_r2.json"),config)
    target=root/f"candidates_r{args.round}.jsonl"
    complete={}
    if target.exists():
        complete={r["sound_asset_id"]:r for r in map(json.loads,target.read_text().splitlines())}
        for r in complete.values():
            assert sha(r["path"]) == r["sha256"]
    tasks=[r for r in tasks if r["sound_asset_id"] not in complete]
    batch_size=args.batch_size;oom_retried=False;index=0;gpu_start=time.time()
    with target.open("a") as stream:
        while index<len(tasks):
            if time.time()+60 >= args.deadline_utc:
                print("DEADLINE_STOP",index,len(tasks),flush=True);break
            batch=tasks[index:index+batch_size]
            seed=202610120 + args.round*100000 + index*97
            duration=max(r["duration_s"] for r in batch)
            align=model.pretransform.downsampling_ratio
            sample_size=mc["sample_size"] if args.round == 2 else math.ceil((duration+1)*44100/align)*align
            cond=[{"prompt":r["prompt"],"seconds_start":0,"seconds_total":r["duration_s"]} for r in batch]
            before=time.time()
            try:
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                    output=generate_diffusion_cond(model,steps=100,cfg_scale=7,
                        conditioning=cond,batch_size=len(batch),sample_size=sample_size,
                        seed=seed,sigma_min=0.3,sigma_max=500,
                        sampler_type="dpmpp-3m-sde",device="cuda")
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                if oom_retried or batch_size==1:raise
                oom_retried=True;batch_size=1;torch.cuda.empty_cache()
                progress(root,"显存不足：批量 2 降至 1，重试一次。")
                continue
            elapsed=time.time()-before
            output=output.to(torch.float32).cpu().numpy()
            assert output.shape[1]==2
            for bi,(r,y) in enumerate(zip(batch,output)):
                n=round(r["duration_s"]*44100)
                stereo=y[:,:n].T
                if stereo.shape!=(n,2) or not np.isfinite(stereo).all():
                    raise RuntimeError("nonfinite or short stereo output")
                peak=float(np.max(np.abs(stereo)))
                if peak>0:stereo=stereo/peak
                stereo_path=root/"stereo_44100"/r["sound_class"]/(r["sound_asset_id"]+".wav")
                mono_path=root/"mono_16000"/r["sound_class"]/(r["sound_asset_id"]+".wav")
                stereo_path.parent.mkdir(parents=True,exist_ok=True);mono_path.parent.mkdir(parents=True,exist_ok=True)
                if stereo_path.exists() or mono_path.exists():raise RuntimeError("candidate path already exists")
                sf.write(stereo_path,stereo,44100,subtype="FLOAT")
                # Channel averaging occurs before resampling.
                mono=resample_poly(stereo.mean(axis=1),160,441)
                sf.write(mono_path,mono,16000,subtype="FLOAT")
                result=dict(r,path=str(mono_path),sha256=sha(mono_path),
                    stereo_path=str(stereo_path),stereo_sha256=sha(stereo_path),
                    seed=seed,seed_batch_index=bi,batch_conditioning=cond,
                    generation=dict(model_id=config["model_id"],model_sha256=digest,
                        model_config_sha256=sha(args.model_dir/"model_config.json"),
                        text_encoder="t5-base",text_encoder_path=str(t5_path),
                        stable_audio_tools_version=importlib.metadata.version("stable-audio-tools"),
                        steps=100,cfg_scale=7,sampler_type="dpmpp-3m-sde",sigma_min=0.3,
                        sigma_max=500,sample_size=sample_size,seed=seed,seed_batch_index=bi,
                        batch_size=len(batch),batch_conditioning=cond,
                        prompt_id=r["prompt_id"],prompt=r["prompt"],duration_s=r["duration_s"],
                        precision="cuda autocast float16",stereo_peak_before_normalization=peak,
                        sample_rate_before=44100,channels_before=2,sample_rate_after=16000,
                        channels_after=1,conversion="channel mean then scipy resample_poly(160,441)",
                        batch_wall_seconds=elapsed),
                    label_provenance="generated_stable_audio_open_1.0",
                    source_license=config["license"],owner_acceptance=config["owner_acceptance"],
                    allowed_splits=["train"],split="train",is_program_synthesized=True,
                    source_dataset="Stable Audio Open 1.0",source_type="generated_audio",
                    source_uri="https://huggingface.co/stabilityai/stable-audio-open-1.0",
                    human_listening_used=False)
                stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
                complete[r["sound_asset_id"]]=result
            del output
            index+=len(batch)
            if index%12<batch_size or index==len(tasks):
                print("GENERATED",len(complete),"batch_seconds",round(elapsed,2),flush=True)
    receipt=dict(round=args.round,generated_count=len(complete),seconds=time.time()-start,
        gpu_generation_wall_seconds=time.time()-gpu_start,model_loading_wall_seconds=gpu_start-start,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        oom_batch_reduction=oom_retried,complete=len(complete)==len(candidate_plan(config,args.round,set(args.classes))),
        human_listening_used=False)
    dump(root/f"generation_receipt_r{args.round}.json",receipt)
    progress(root,f"第 {args.round} 轮生成结束：{len(complete)} 个候选；完整={receipt['complete']}。")
    print("DONE",json.dumps(receipt),flush=True)
if __name__=="__main__":main()
