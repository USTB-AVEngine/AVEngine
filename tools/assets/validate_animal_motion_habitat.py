"""Run one dog or cat through actual Habitat, native floor and posed geometry."""
import argparse,json
from pathlib import Path
from avengine.assets.animal_motion.native import run_habitat

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--asset',required=True);p.add_argument('--house',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--route',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--speed',type=float,default=.08);p.add_argument('--period',type=float,default=2.4);p.add_argument('--seconds',type=float,default=4.8);p.add_argument('--mode',choices=['walk_animation','slide_no_animation'],default='walk_animation');p.add_argument('--physical-controls',action='store_true',help='Record real 180/90 degree drawable negatives in the existing native context');a=p.parse_args()
    record=next(x for x in json.loads(a.inventory.read_text())['assets'] if x['asset_id']==a.asset)
    result=run_habitat(record,json.loads(a.house.read_text()),json.loads(a.runtime.read_text()),json.loads(a.route.read_text()),a.output,speed=a.speed,period=a.period,seconds=a.seconds,mode=a.mode,physical_controls=a.physical_controls)
    print({k:v for k,v in result.items() if k not in ('frames','checks','renderer','stage_transform','source_provenance','direction_negative_controls','native_direction_negative_frames','support_plane')},flush=True)
    if result['status']!='geometry_pass_audio_pending':raise SystemExit(2)
if __name__=='__main__':main()
