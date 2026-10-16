"""Remove ISNet exterior islands and boundary wood halos, preserving first receipts."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label,binary_dilation
from v7_animal_variants_color import rgb_to_lab

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);a=p.parse_args();rows=json.loads((a.task_root/'evidence/segmentation_results.json').read_text())
    for row in rows:
        if row['status']=='pass':continue
        d=a.task_root/'assets'/row['asset_id']/'reference';original=json.loads((d/'segmentation_receipt.json').read_text());shutil.copy2(d/'segmentation_receipt.json',d/'segmentation_receipt_first.json');shutil.copy2(d/'mask.png',d/'mask_first.png');shutil.copy2(d/'input_rgba_isnet.png',d/'input_rgba_isnet_first.png');rgb=np.array(Image.open(d/'candidate.png').convert('RGB'));alpha=np.array(Image.open(d/'mask.png'));components,n=label(alpha>=128);sizes=np.bincount(components.ravel());sizes[0]=0;largest=components==sizes.argmax();lab=rgb_to_lab(rgb);h=np.mod(np.degrees(np.arctan2(lab[:,:,2],lab[:,:,1])),360);c=np.hypot(lab[:,:,1],lab[:,:,2]);yy,xx=np.indices(largest.shape);wood=(h>=45)&(h<=110)&(c>8)&(yy>rgb.shape[0]*.60);wood=binary_dilation(wood,iterations=2);keep=largest&~wood;components,n=label(keep);sizes=np.bincount(components.ravel());sizes[0]=0;keep=components==sizes.argmax();new_alpha=np.where(keep,alpha,0).astype(np.uint8);rgba=Image.fromarray(rgb).convert('RGBA');rgba.putalpha(Image.fromarray(new_alpha));rgba.save(d/'input_rgba_isnet.png');Image.fromarray(new_alpha).save(d/'mask.png');y,x=np.where(new_alpha>=128);bbox=[int(x.min()),int(y.min()),int(x.max()),int(y.max())];margins=[bbox[0],bbox[1],rgb.shape[1]-1-bbox[2],rgb.shape[0]-1-bbox[3]];row.update(rgba_sha256=hashlib.sha256((d/'input_rgba_isnet.png').read_bytes()).hexdigest(),mask_sha256=hashlib.sha256((d/'mask.png').read_bytes()).hexdigest(),bbox_xyxy=bbox,margins_ltrb=margins,foreground_fraction=float(keep.mean()),status='pass' if min(margins)>8 and .03<keep.mean()<.7 else 'fail',postprocessing={'method':'largest ISNet component; remove lower-frame wood hue 45..110, chroma>8 with 2px antialias halo; largest remaining component','original_isnet_receipt':str(d/'segmentation_receipt_first.json'),'original_rgba_sha256':original['rgba_sha256'],'removed_foreground_pixels':int((alpha>=128).sum()-keep.sum()),'affects':'exterior islands and silent generated perch; no anatomy acceptance claimed'});(d/'segmentation_receipt.json').write_text(json.dumps(row,indent=2)+'\n');print(row['asset_id'],row['status'],margins)
    (a.task_root/'evidence/segmentation_results.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
