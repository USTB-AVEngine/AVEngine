"""CPU ISNet cut-out of this task's pet-bird candidate."""
import argparse,json,os,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
def main(T):
 R=T/'round2';D=R/'assets/bird_2d';src=D/'candidate.png';weight=Path('/data/models/rembg/isnet-general-use/isnet-general-use.onnx')
 opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
 sess=ort.InferenceSession(str(weight),sess_options=opt,providers=['CPUExecutionProvider'])
 rgb=Image.open(src).convert('RGB');a=np.asarray(rgb.resize((1024,1024),Image.Resampling.LANCZOS),dtype=np.float32);a=a/max(float(a.max()),1);a=(a-.5).transpose(2,0,1)[None]
 pred=np.asarray(sess.run(None,{sess.get_inputs()[0].name:a})[0])[0,0];pred=(pred-pred.min())/max(float(pred.max()-pred.min()),1e-9)
 alpha=Image.fromarray(np.uint8(np.clip(pred,0,1)*255)).resize(rgb.size,Image.Resampling.LANCZOS)
 rgba=rgb.convert('RGBA');rgba.putalpha(alpha);rgba.save(D/'input_rgba_isnet.png');alpha.save(D/'mask.png')
 data={'schema':'avengine_v7_bird_isnet_v1','source':str(src),'output_rgba':str(D/'input_rgba_isnet.png'),'checkpoint':str(weight),'checkpoint_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'onnxruntime_providers':sess.get_providers(),'foreground_fraction_alpha_ge_128':float(np.mean(np.asarray(alpha)>=128)),'alpha_range':[int(np.asarray(alpha).min()),int(np.asarray(alpha).max())],'mask_erosion':False,'pixal3d_status':'not_run_pending_project_owner_image_review','source_point_semantic':'bird beak; metric 3D point not available until mesh exists'}
 (D/'segmentation.json').write_text(json.dumps(data,indent=2));print(json.dumps(data))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);main(p.parse_args().task_root)
