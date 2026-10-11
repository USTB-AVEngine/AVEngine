"""CPU ISNet segmentation and image-boundary checks for generated bird references."""
import argparse,json,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import label
def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True);t0=time.time()
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
    sess=ort.InferenceSession(str(a.checkpoint),sess_options=opt,providers=["CPUExecutionProvider"])
    rgb=Image.open(a.input).convert("RGB");x=np.asarray(rgb.resize((1024,1024),Image.Resampling.LANCZOS),dtype=np.float32);x=x/max(float(x.max()),1);x=(x-.5).transpose(2,0,1)[None]
    pred=np.asarray(sess.run(None,{sess.get_inputs()[0].name:x})[0])[0,0];pred=(pred-pred.min())/max(float(pred.max()-pred.min()),1e-9)
    alpha=Image.fromarray(np.uint8(np.clip(pred,0,1)*255)).resize(rgb.size,Image.Resampling.LANCZOS);rgba=rgb.convert("RGBA");rgba.putalpha(alpha);rgba.save(a.output_dir/"input_rgba_isnet.png");alpha.save(a.output_dir/"mask.png")
    mask=np.asarray(alpha)>=128;yy,xx=np.where(mask);components,n=label(mask);sizes=np.bincount(components.ravel())[1:];major=sorted([int(v) for v in sizes if v>=max(50,int(mask.sum()*.005))],reverse=True)
    bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None
    margins=[bbox[0],bbox[1],rgb.width-1-bbox[2],rgb.height-1-bbox[3]] if bbox else None
    data={"input":str(a.input),"output":str(a.output_dir/"input_rgba_isnet.png"),"checkpoint":str(a.checkpoint),"providers":sess.get_providers(),"resolution":list(rgb.size),"foreground_pixels":int(mask.sum()),"foreground_fraction":float(mask.mean()),"foreground_bbox_xyxy":bbox,"margin_pixels_ltrb":margins,"major_component_sizes_pixels":major,"complete_inside_image":bool(margins and min(margins)>8),"no_mask_erosion":True,"automatic_anatomy_status":"not_claimed; segmentation topology is not an anatomical detector","seconds":time.time()-t0}
    (a.output_dir/"segmentation.json").write_text(json.dumps(data,indent=2)+"\n");print(json.dumps(data),flush=True)
if __name__=="__main__":main()
