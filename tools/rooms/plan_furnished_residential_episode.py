#!/usr/bin/env python3
"""Plan a static furnished residential episode through the AVEngine room API."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
from avengine.rooms.furnished_episode import (
    _actor_state,
    _overview_target_bounds,
    _camera_for_runtime,
    _load_json,
    _load_static_triangle_geometry,
    build_episode_plan,
    plan_furnished_residential_episode,
    reuse_camera_from_plan,
)

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--room","--room-manifest",dest="room",type=Path,required=True)
    p.add_argument("--asset-root",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--activity",default="seated")
    p.add_argument("--pose-bindings",type=Path)
    p.add_argument("--pose-request",type=Path)
    p.add_argument("--map-path");p.add_argument("--scene-id")
    p.add_argument("--seat-count",type=int,default=2);p.add_argument("--actor-count",type=int,default=2)
    p.add_argument("--frame-count",type=int,default=75);p.add_argument("--frame-rate-hz",type=float,default=15.0)
    p.add_argument("--sample-rate-hz",type=int,default=16000);p.add_argument("--grid-step-m",type=float,default=2.0)
    p.add_argument("--camera-height-m",type=float,default=1.55);p.add_argument("--camera-source-plan",type=Path)
    p.add_argument("--overview-only",action="store_true")
    return p.parse_args()

def main() -> int:
    a=parse_args()
    plan=plan_furnished_residential_episode(**vars(a))
    print(json.dumps({"output":str(a.output.expanduser().resolve()),"room_id":plan["room_layout"]["room_id"],"camera_candidates":len(plan["camera_candidates"]["candidates"]),"seats":plan["seat_layout"]["selected_seat_ids"],"native_validation_status":"not_run"},ensure_ascii=False))
    return 0
if __name__=="__main__": raise SystemExit(main())
