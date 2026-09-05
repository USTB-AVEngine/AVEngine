#!/usr/bin/env python3
"""Plan and render one static-seated furnished room as a Studio research task."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
from avengine.rooms.furnished_episode import plan_furnished_residential_episode

def parse_args():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--room",type=Path,required=True);p.add_argument("--asset-root",type=Path)
 p.add_argument("--pose-bindings",type=Path,required=True);p.add_argument("--pose-request",type=Path)
 p.add_argument("--activity",required=True);p.add_argument("--map-path",required=True)
 p.add_argument("--camera-source-plan",type=Path);p.add_argument("--seat-count",type=int,default=4);p.add_argument("--actor-count",type=int,default=4)
 p.add_argument("--frame-count",type=int,default=75);p.add_argument("--frame-rate-hz",type=float,default=15.0);p.add_argument("--sample-rate-hz",type=int,default=16000)
 p.add_argument("--grid-step-m",type=float,default=2.0);p.add_argument("--camera-height-m",type=float,default=1.55)
 p.add_argument("--spear-ext-dir",type=Path);p.add_argument("--uproject",type=Path,required=True);p.add_argument("--unreal-editor",type=Path,required=True)
 p.add_argument("--rpc-port",type=int,default=39379);p.add_argument("--graphics-adapter",type=int,default=0)
 p.add_argument("--width",type=int,default=1280);p.add_argument("--height",type=int,default=720);p.add_argument("--exposure-bias-ev",type=float)
 p.add_argument("--streaming-warmup-frames",type=int,default=180);p.add_argument("--native-multimodal",action="store_true");p.add_argument("--keep-frames",action="store_true")
 p.add_argument("--output",type=Path,required=True)
 return p.parse_args()

def run(a):
 output=a.output.expanduser().resolve()
 if output.exists(): raise FileExistsError(f"refusing to replace existing output: {output}")
 output.mkdir(parents=True)
 plan_root=output/"plan";capture_root=output/"capture"
 plan_furnished_residential_episode(room=a.room,asset_root=a.asset_root,pose_bindings=a.pose_bindings,pose_request=a.pose_request,output=plan_root,activity=a.activity,map_path=a.map_path,seat_count=a.seat_count,actor_count=a.actor_count,frame_count=a.frame_count,frame_rate_hz=a.frame_rate_hz,sample_rate_hz=a.sample_rate_hz,grid_step_m=a.grid_step_m,camera_height_m=a.camera_height_m,camera_source_plan=a.camera_source_plan)
 runner=Path(__file__).resolve().parents[1]/"rooms"/"run_spear_residential_episode.py"
 cmd=[sys.executable,str(runner),"--episode-root",str(plan_root),"--uproject",str(a.uproject),"--unreal-editor",str(a.unreal_editor),"--output",str(capture_root),"--rpc-port",str(a.rpc_port),"--graphics-adapter",str(a.graphics_adapter),"--width",str(a.width),"--height",str(a.height),"--streaming-warmup-frames",str(a.streaming_warmup_frames),"--visual-only-research"]
 if a.spear_ext_dir: cmd += ["--spear-ext-dir",str(a.spear_ext_dir)]
 if a.exposure_bias_ev is not None: cmd += ["--exposure-bias-ev",str(a.exposure_bias_ev)]
 if a.native_multimodal: cmd.append("--native-multimodal")
 if a.keep_frames: cmd.append("--keep-frames")
 subprocess.run(cmd,check=True)
 receipt={"status":"research_only","activity":"seated","plan_root":str(plan_root),"capture_root":str(capture_root),"executor":"tools/rooms/run_spear_residential_episode.py","qualification_claim":False}
 (output/"studio_seated_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
 return receipt
def main(): print(json.dumps(run(parse_args()),indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
