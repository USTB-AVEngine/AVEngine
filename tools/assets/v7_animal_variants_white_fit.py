"""Fit a neutral white boundary only on known native human-shirt calibration pixels."""
import json,hashlib,sys,datetime
from pathlib import Path
import numpy as np
from PIL import Image
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W/'tools/assets'))
from v7_animal_variants_color import measure
def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--task-root',type=Path,required=True);args=parser.parse_args();T=args.task_root;p=T/'evidence/color_calibration.json';cal=json.loads(p.read_text());old_sha=hashlib.sha256(p.read_bytes()).hexdigest();(T/'evidence/color_calibration_before_white_fit.json').write_bytes(p.read_bytes());loaded=[(r,np.array(Image.open(r['rgb_path']).convert('RGB')),np.array(Image.open(r['mask_path']).convert('L'))>0) for r in cal['measurements']];options=[]
    for white in [35.,40.,45.,50.,55.,60.]:
     threshold={**cal['thresholds'],'white_l_min':white};fit=[(r,measure(rgb,mask,threshold)) for r,rgb,mask in loaded if r['split']=='calibration'];fractions=[m['family_fractions'][r['expected_label']] for r,m in fit];correct=sum(m['status']=='pass' and m['measured_label']==r['expected_label'] for r,m in fit);options.append({'white_l_min':white,'fitting_correct':correct,'fitting_count':len(fit),'minimum_expected_family_fraction':min(fractions),'mean_expected_family_fraction':float(np.mean(fractions))})
    best=max(options,key=lambda r:(r['fitting_correct'],r['minimum_expected_family_fraction'],r['mean_expected_family_fraction']));threshold={**cal['thresholds'],'white_l_min':best['white_l_min']};rows=[{**r,**measure(rgb,mask,threshold),'correct':measure(rgb,mask,threshold)['status']=='pass' and measure(rgb,mask,threshold)['measured_label']==r['expected_label']} for r,rgb,mask in loaded];groups={s:{'count':sum(r['split']==s for r in rows),'correct':sum(r['split']==s and r['correct'] for r in rows)} for s in ['calibration','held_out']};cal.update(thresholds=threshold,measurements=rows,groups=groups,status='pass' if all(r['count']==r['correct'] for r in groups.values()) else 'fail',white_boundary_refit={'method':'select on human-shirt calibration split only, lexicographic correct count / worst known-family fraction / mean known-family fraction; all other boundaries frozen; held-out validation only after selection','fitting_candidates':options,'selected':best,'previous_calibration_sha256':old_sha,'animal_pixels_used_for_fitting':False,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});assert cal['status']=='pass';p.write_text(json.dumps(cal,ensure_ascii=False,indent=2)+'\n');print('WHITE_FIT',threshold,groups)

if __name__=='__main__':main()
