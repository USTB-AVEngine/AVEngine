"""Run a resident CPU ISNet session for object cutouts, optionally following a bounded FLUX batch."""
import argparse,datetime,json,time
from pathlib import Path
import numpy as np
from PIL import Image

def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--follow-job');a=p.parse_args();T=a.task_root
    import onnxruntime as ort
    opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1;sess=ort.InferenceSession(str(a.checkpoint),sess_options=opt,providers=['CPUExecutionProvider']);records={}
    while True:
        for src in sorted((T/'new').glob('*/candidate.png')):
            aid=src.parent.name
            if aid in records:continue
            flux=src.with_name('flux.json')
            if not flux.exists() or json.loads(flux.read_text()).get('status')!='pass':continue
            dst=src.with_name('input_rgba.png');proof=src.with_name('segmentation.json')
            if dst.exists() and proof.exists():records[aid]=json.loads(proof.read_text());continue
            start=time.monotonic();rgb=Image.open(src).convert('RGB');array=np.asarray(rgb.resize((1024,1024),Image.Resampling.LANCZOS),dtype=np.float32)/255;array=(array-.5).transpose(2,0,1)[None];pred=np.asarray(sess.run(None,{sess.get_inputs()[0].name:array})[0])[0,0];pred=(pred-pred.min())/max(float(pred.max()-pred.min()),1e-9);pred=np.where(pred<.5,0,pred);alpha=Image.fromarray(np.uint8(np.clip(pred,0,1)*255)).resize(rgb.size,Image.Resampling.LANCZOS);coverage=float(np.mean(np.asarray(alpha)>127));rgba=rgb.convert('RGBA');rgba.putalpha(alpha);rgba.save(dst);rec={'status':'pass' if .03<=coverage<=.80 else 'fail','asset_id':aid,'source':str(src),'output_rgba':str(dst),'checkpoint':str(a.checkpoint),'foreground_share':coverage,'alpha_threshold':.5,'resident_session':True,'providers':sess.get_providers(),'wall_seconds':time.monotonic()-start};dump(proof,rec);records[aid]=rec;dump(T/'segmentation_results.json',list(records.values()));print('CUTOUT',aid,rec['status'],coverage,flush=True)
        dump(T/'segmentation_results.json',list(records.values()))
        if not a.follow_job:break
        jobs=json.loads((T/'JOBS.json').read_text());job=next((j for j in jobs if j['name']==a.follow_job),None)
        if job is None or job['status'] not in ['running','waiting_gpu','starting']:break
        budget=json.loads((T/'budget.json').read_text())
        if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(budget['gpu_generation_cutoff_utc']):break
        time.sleep(5)
if __name__=='__main__':main()
