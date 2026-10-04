"""Raw Habitat RGB/depth/semantic comparison; never compare compressed videos."""
import json,hashlib
from pathlib import Path
import numpy as np
from resources import write_json
def compare(a,b,out,label):
 rows=[]
 for index in (0,149,300,449):
  x=np.load(Path(a)/f'frame_{index:04d}.npz');y=np.load(Path(b)/f'frame_{index:04d}.npz')
  for name in ('rgb','depth','semantic'):
   p=x[name];q=y[name]
   if p.shape!=q.shape or p.dtype!=q.dtype:raise ValueError('reference shape/dtype differs')
   diff=np.abs(p.astype(np.float64)-q.astype(np.float64));different=p!=q
   row={'frame_index':index,'channel':name,'shape':list(p.shape),'dtype':str(p.dtype),'exact_equal':bool(np.array_equal(p,q)),'mean_absolute_difference':float(diff.mean()),'maximum_absolute_difference':float(diff.max()),'different_elements_fraction':float(different.mean()),'different_pixels_fraction':float(np.any(different,axis=-1).mean()) if name=='rgb' else float(different.mean()),'left_raw_sha256':hashlib.sha256(p.tobytes()).hexdigest(),'right_raw_sha256':hashlib.sha256(q.tobytes()).hexdigest(),'unit':'uint8 channel value' if name=='rgb' else ('metres' if name=='depth' else 'semantic ID')}
   rows.append(row)
 result={'comparison':label,'left':str(a),'right':str(b),'frames':4,'raw_arrays':12,'all_exact_equal':all(r['exact_equal'] for r in rows),'rows':rows,'compression_comparison':False,'production_admission':False};write_json(out,result);return result
