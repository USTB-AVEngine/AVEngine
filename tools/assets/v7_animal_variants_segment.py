"""Batch CPU ISNet matting; wait for owned FLUX receipts, with a finite deadline."""
import argparse,hashlib,json,time,shutil
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import label

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--only-missing',action='store_true');p.add_argument('--result-name',default='segmentation_results.json');a=p.parse_args();items=[x for x in json.loads((a.task_root/'plan.json').read_text())['animals'] if x['mode']=='generated_static'];opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;t0=time.time();sess=ort.InferenceSession(str(a.checkpoint),sess_options=opt,providers=['CPUExecutionProvider']);model_sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();records=[]
    for item in items:
        if a.only_missing and (a.task_root/'assets'/item['asset_id']/'reference/segmentation_receipt.json').exists():continue
        d=a.task_root/'assets'/item['asset_id']/'reference';input=d/'candidate.png'
        while not (d/'generation_receipt.json').exists():
            if time.time()-t0>1800:raise RuntimeError('reference receipt timed out: '+item['asset_id'])
            time.sleep(3)
        t=time.time();rgb=Image.open(input).convert('RGB');x=np.asarray(rgb.resize((1024,1024),Image.Resampling.LANCZOS),dtype=np.float32);x=x/max(float(x.max()),1);x=(x-.5).transpose(2,0,1)[None];pred=np.asarray(sess.run(None,{sess.get_inputs()[0].name:x})[0])[0,0];pred=(pred-pred.min())/max(float(pred.max()-pred.min()),1e-9);alpha=Image.fromarray(np.uint8(np.clip(pred,0,1)*255)).resize(rgb.size,Image.Resampling.LANCZOS);rgba=rgb.convert('RGBA');rgba.putalpha(alpha);rgba.save(d/'input_rgba_isnet.png');alpha.save(d/'mask.png');mask=np.asarray(alpha)>=128;yy,xx=np.where(mask);components,n=label(mask);sizes=np.bincount(components.ravel())[1:];bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None;margins=[bbox[0],bbox[1],rgb.width-1-bbox[2],rgb.height-1-bbox[3]] if bbox else None
        rec={'asset_id':item['asset_id'],'checkpoint':str(a.checkpoint),'checkpoint_sha256':model_sha,'weight_license':'Apache-2.0','code_license':'MIT / task-authored ONNX preprocessing','providers':sess.get_providers(),'input_sha256':hashlib.sha256(input.read_bytes()).hexdigest(),'rgba_sha256':hashlib.sha256((d/'input_rgba_isnet.png').read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256((d/'mask.png').read_bytes()).hexdigest(),'foreground_fraction':float(mask.mean()),'bbox_xyxy':bbox,'margins_ltrb':margins,'major_component_sizes':sorted([int(v) for v in sizes if v>=max(50,int(mask.sum()*.005))],reverse=True),'status':'pass' if margins and min(margins)>8 and .03<mask.mean()<.7 else 'fail','seconds':time.time()-t,'anatomy_status':'not_claimed_from_segmentation'};(d/'segmentation_receipt.json').write_text(json.dumps(rec,indent=2)+'\n');records.append(rec);(a.task_root/'evidence'/a.result_name).write_text(json.dumps(records,indent=2)+'\n');print('ISNET_DONE',item['asset_id'],rec['status'],flush=True)
    (a.task_root/'evidence'/(a.result_name.replace('results','complete'))).write_text(json.dumps({'count':len(records),'seconds':time.time()-t0})+'\n')
if __name__=='__main__':main()
