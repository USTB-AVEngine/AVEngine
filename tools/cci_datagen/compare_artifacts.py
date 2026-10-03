#!/usr/bin/env python3
"""Compare actual FOA samples, decoded video frames and exact QA JSON content."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from resources import write_json

def digest(b):return hashlib.sha256(b).hexdigest()
def audio(p):
 x,sr=sf.read(p,dtype='float32',always_2d=True)
 if x.shape[1]!=4:raise ValueError('FOA must contain four channels')
 return {'sample_rate':sr,'shape':list(x.shape),'sample_sha256':digest(np.asarray(x,dtype='<f4').tobytes()),'file_sha256':digest(Path(p).read_bytes())}
def video(p):
 result=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-map','0:v:0','-f','framemd5','-'],capture_output=True,text=True,check=True)
 frames=[x for x in result.stdout.splitlines() if x and not x.startswith('#')]
 return {'frame_count':len(frames),'decoded_frame_hashes':frames,'all_frames_sha256':digest('\n'.join(frames).encode())}
def questions(p):
 data=json.loads(Path(p).read_text());return {'sha256':digest(json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())}
p=argparse.ArgumentParser();p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--output',required=True);a=p.parse_args()
# Input JSON is {world_id: {foa: WAV, video: MP4, questions: JSON}}.
l=json.loads(Path(a.left).read_text());r=json.loads(Path(a.right).read_text());result={}
for world in sorted(set(l)|set(r)):
 checks={}
 for key,fn in [('foa',audio),('video',video),('questions',questions)]:
  if world not in l or world not in r or key not in l[world] or key not in r[world]:checks[key]={'status':'not_run','reason':'missing actual modality artifact'};continue
  left,right=fn(l[world][key]),fn(r[world][key]); comparable='sample_sha256' if key=='foa' else 'all_frames_sha256' if key=='video' else 'sha256'
  checks[key]={'status':'pass' if left[comparable]==right[comparable] and (key!='foa' or (left['shape']==right['shape'] and left['sample_rate']==right['sample_rate'])) else 'fail','left':left,'right':right}
 result[world]=checks
passed=bool(result) and all(x['status']=='pass' for v in result.values() for x in v.values())
write_json(a.output,{'status':'pass' if passed else 'incomplete_or_different','worlds':result});raise SystemExit(0 if passed else 1)
