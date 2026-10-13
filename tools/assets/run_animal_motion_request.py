"""Execute Walking first; qualify an explicitly marked standing slide if needed."""
import argparse,json
from pathlib import Path
from avengine.assets.animal_motion.service import execute_request

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--asset',required=True);p.add_argument('--house',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--route',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse',type=Path);p.add_argument('--speed',type=float,default=.05);p.add_argument('--seconds',type=float,default=4.8);p.add_argument('--ray-multiplier',type=int,default=27);a=p.parse_args()
 record=next(r for r in json.loads(a.inventory.read_text())['assets'] if r['asset_id']==a.asset)
 result=execute_request(record,json.loads(a.house.read_text()),json.loads(a.runtime.read_text()),json.loads(a.route.read_text()),a.output,speed=a.speed,seconds=a.seconds,ray_multiplier=a.ray_multiplier,reuse=json.loads(a.reuse.read_text()) if a.reuse else None)
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
