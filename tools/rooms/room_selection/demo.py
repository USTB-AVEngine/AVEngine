"""Build a descriptive HM3D/MP3D demo using the frozen merged protocol.

MP3D measurements call process_house, not a second decision implementation.
HM3D imports existing held-out results and human records without reevaluation.
All writes go to the explicit output. External assets and human reviews are
read-only. GPU rendering is a separate, explicitly selected-device command.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, platform, subprocess, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from . import PROTOCOL_VERSION
from .run import load_parameters, process_house, write_json, now
from .mp3d import read_house
from .geometry import infer_type


def register(args):
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    if (out/"inputs.json").exists(): raise FileExistsError("demo already registered")
    spec,p=load_parameters(args.thresholds)
    selected=json.loads(args.hm3d_selection.read_text())
    holdout=set(json.loads((args.merged_results/"house_analysis_split.json").read_text())["holdout"])
    hm3d=[]
    for h in selected["houses"]:
        house=h["house"]
        if house not in holdout: raise ValueError("HM3D example is not held out: "+house)
        source=args.merged_results/"houses"/(house+".json")
        data=json.loads(source.read_text())
        if data["status"]!="measured": raise ValueError(source)
        data["demo_source"]=str(source)
        data["demo_policy"]="unchanged original merged measurement, automatic decisions and smy human records; no reevaluation"
        write_json(out/"hm3d_houses"/(house+".json"),data)
        hm3d.append(dict(**h,source=str(source)))
    if len(set(args.scene_id))!=6 or len(hm3d)!=6: raise ValueError("six distinct houses per family required")
    mp3d=[]; rows=[]
    for sid in args.scene_id:
        scene=(args.mp3d_root/sid).resolve(strict=True)
        if scene.parent!=args.mp3d_root.resolve(strict=True): raise ValueError("scene root escape")
        ann=scene/(sid+".house"); sem=scene/(sid+"_semantic.ply"); glb=scene/(sid+".glb"); nav=scene/(sid+".navmesh")
        for asset in [ann,sem,glb,nav]:
            if not asset.is_file(): raise FileNotFoundError(asset)
        instances,regions,levels=read_house(ann)
        house="mp3d_"+sid; house_rows=[]
        for rid,native in sorted(regions.items()):
            members={i:v for i,v in instances.items() if v["region_id"]==rid}
            room_type,evidence=infer_type(members,native["region_label"])
            row=dict(schema=PROTOCOL_VERSION,family="mp3d",house=house,room_label="R"+str(rid),region_id=rid,
                registered_at_sgt=now(),semantic_region_present=True,semantic_instance_count=len(members),semantic_categories=sorted({m["category"] for m in members.values()}),
                native_region=native,legacy_room=None,legacy_rooms_source=None,human=None,
                stage0=dict(status="registered",reason_codes=[]),stage1=dict(status="not_run",reason_codes=[]),stage2=dict(status="not_run",reason_codes=[]),
                stage4=dict(status="not_run",reason_codes=["MP3D_FORMAL_HOUSE_AUDIOVISUAL_GATE_NOT_AVAILABLE"]),
                scene_directory=str(scene),semantic_source=str(sem),annotation_source=str(ann),navmesh_source=str(nav),metrics=None,
                room_type=room_type,room_type_evidence=evidence)
            house_rows.append(row);rows.append(row)
        job=dict(house=house,family="mp3d",scene_directory=str(scene),navmesh=str(nav),rows=house_rows)
        write_json(out/"jobs"/(house+".json"),job)
        mp3d.append(dict(house=house,id=sid,region_count=len(regions),native_levels=sorted(levels),scene_directory=str(scene)))
    threshold_bytes=args.thresholds.read_bytes()
    (out/"frozen_thresholds.reference.yaml").write_bytes(threshold_bytes)
    inputs=dict(created_at_sgt=now(),machine=platform.node(),baseline_commit="b2d625ff32258bd7fb19fdcc17764a752d774034",threshold_source=str(args.thresholds.resolve()),threshold_sha256=hashlib.sha256(threshold_bytes).hexdigest(),threshold_spec=spec,parameters=p,
        merged_results=str(args.merged_results.resolve()),hm3d=hm3d,mp3d=mp3d,mp3d_human_labels=None,mp3d_accuracy="not computed; no human labels",selection_policy="descriptive demonstration; purposive size/storey and disagreement coverage, not a random benchmark")
    write_json(out/"inputs.json",inputs)
    with (out/"rooms_registry.stage0.jsonl").open("w") as stream:
        for row in rows: stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    print("DEMO_REGISTERED",json.dumps(dict(hm3d_houses=len(hm3d),mp3d_houses=len(mp3d),mp3d_regions=len(rows))),flush=True)


def measure(args):
    inputs=json.loads((args.output/"inputs.json").read_text())
    if hashlib.sha256(Path(inputs["threshold_source"]).read_bytes()).hexdigest()!=inputs["threshold_sha256"]: raise ValueError("frozen thresholds changed")
    receipt=dict(status="running",started_at_sgt=now(),controller_pid=os.getpid(),machine=platform.node(),cpu_only=True,workers=args.workers,commands={})
    write_json(args.output/"measurement_receipt.json",receipt)
    receipt_lock = threading.Lock()
    (args.output/"logs").mkdir(exist_ok=True)
    def one(h):
        house=h["house"]; dest=args.output/"houses"/(house+".json")
        if dest.exists():
            d=json.loads(dest.read_text())
            if d.get("status")=="measured": return dict(house=house,status="cached",seconds=d["seconds"])
        cmd=[sys.executable,"-u","-m","tools.rooms.room_selection.demo","worker","--output",str(args.output),"--house",house]
        with (args.output/"logs"/(house+".log")).open("a") as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            with receipt_lock:
                receipt["commands"][house]=dict(pid=proc.pid,command=cmd,log=str(args.output/"logs"/(house+".log")))
                write_json(args.output/"measurement_receipt.json",receipt)
            try: rc=proc.wait(timeout=args.house_timeout)
            except subprocess.TimeoutExpired:
                proc.terminate();proc.wait();rc=-15
        if rc: return dict(house=house,status="failed",exit_code=rc)
        d=json.loads(dest.read_text()); return dict(house=house,status=d["status"],seconds=d["seconds"])
    with ThreadPoolExecutor(max_workers=args.workers) as pool: results=list(pool.map(one,inputs["mp3d"]))
    receipt.update(status="complete" if all(r["status"] in {"cached","measured"} for r in results) else "incomplete",finished_at_sgt=now(),houses=results)
    write_json(args.output/"measurement_receipt.json",receipt)
    print("DEMO_MEASURED",json.dumps(receipt,ensure_ascii=False),flush=True)
    if receipt["status"]!="complete": raise RuntimeError("MP3D measurement incomplete")
    with (args.output/"rooms_registry.cpu.jsonl").open("w") as stream:
        for h in inputs["mp3d"]:
            d=json.loads((args.output/"houses"/(h["house"]+".json")).read_text())
            for row in d["rows"]: stream.write(json.dumps(row,ensure_ascii=False)+"\n")


def worker(args):
    inputs=json.loads((args.output/"inputs.json").read_text())
    args.thresholds=Path(inputs["threshold_source"])
    args.skip_splitting=False;args.media_root=Path(os.environ["AVENGINE_ROOM_MEDIA_ROOT"])
    args.overhead_base=None
    job=json.loads((args.output/"jobs"/(args.house+".json")).read_text())
    process_house(args,job)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["register","measure","worker","render","page"])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--thresholds",type=Path)
    parser.add_argument("--mp3d-root",type=Path)
    parser.add_argument("--scene-id",action="append",default=[])
    parser.add_argument("--hm3d-selection",type=Path)
    parser.add_argument("--merged-results",type=Path)
    parser.add_argument("--house");parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--house-timeout",type=int,default=3600)
    parser.add_argument("--gpu",type=int);parser.add_argument("--probe",action="store_true")
    parser.add_argument("--refresh-visuals", action="store_true", help="replace owned demo images/results only; preserve source measurements")
    parser.add_argument("--supplement-only", action="store_true", help="add missed camera-height maps to owned results without remeasurement")
    parser.add_argument("--fit-overviews", action="store_true", help="expand owned display cameras to fit every contour; preserve measurement images")
    args=parser.parse_args()
    if args.stage=="register": register(args)
    elif args.stage=="measure": measure(args)
    elif args.stage=="worker": worker(args)
    elif args.stage=="render":
        if args.gpu is None: parser.error("render requires an explicitly selected --gpu")
        from .demo_render import render
        render(args)
    else:
        from .demo_page import build
        build(args)


if __name__=="__main__":main()
