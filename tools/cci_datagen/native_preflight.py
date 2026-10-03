#!/usr/bin/env python3
"""Separate CPU import and GPU EGL smoke, with structured failure evidence."""
import argparse
import ctypes.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from resources import write_json
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--render',action='store_true');p.add_argument('--scene');p.add_argument('--gpu-device-id',type=int);a=p.parse_args()
base=Path(os.environ['CCI_DATAGEN_BASE']);repo=base/'code';native=base/'native/data/avengine_external'
script='''import json,sys
from avengine.acoustics.runtime import load_habitat_runtime
hs,r=load_habitat_runtime(runtime_mode="current-installed",runtime_prefix=sys.argv[1],magnum_python_site=sys.argv[2],rlr_sdk_root=sys.argv[3])
print(json.dumps({"runtime":r,"rlr_available":hasattr(hs,"RLRAcousticContext")}))
'''
args=[str(native/'runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z'),str(native/'runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages'),str(native/'rlr-sdk/RLRAudioPropagationPkg')]
if a.render:
 if a.gpu_device_id is None:raise ValueError('render probe requires an explicit physical GPU device ID')
 if not a.scene:raise ValueError('render probe requires an explicitly authorized scene')
 script+='''import numpy as np
cfg=hs.SimulatorConfiguration();cfg.scene_id=sys.argv[4];cfg.enable_physics=False;cfg.gpu_device_id=int(sys.argv[5])
sensor=hs.CameraSensorSpec();sensor.uuid="rgb";sensor.sensor_type=hs.SensorType.COLOR;sensor.resolution=[64,64]
agent=hs.agent.AgentConfiguration();agent.sensor_specifications=[sensor]
with hs.Simulator(hs.Configuration(cfg,[agent])) as simulator:
 data=simulator.get_sensor_observations()["rgb"]
 import hashlib
 print(json.dumps({"frame_shape":list(data.shape),"frame_sha256":hashlib.sha256(data.tobytes()).hexdigest()}))
'''
 args.extend([a.scene,str(a.gpu_device_id)])
env=dict(os.environ);env['CUDA_CACHE_PATH']=str(base/'tmp/cuda_cache');env['__GL_SHADER_DISK_CACHE_PATH']=str(base/'tmp/gl_shader_cache');env['PYTHONPATH']=str(repo/'src')+':'+str(repo)
if not a.render:env['CUDA_VISIBLE_DEVICES']=''
start=time.monotonic()
try:
 result=subprocess.run([sys.executable,'-B','-c',script,*args],env=env,text=True,capture_output=True,timeout=30)
 data={'status':'pass' if result.returncode==0 else 'blocked','returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'wall_seconds':time.monotonic()-start,'mode':'egl_render' if a.render else 'cpu_import','libraries':{k:ctypes.util.find_library(k) for k in ['EGL','OpenGL','GLdispatch','cuda']}}
except subprocess.TimeoutExpired as e:
 data={'status':'blocked','reason':'timeout','stdout':str(e.stdout),'stderr':str(e.stderr),'mode':'egl_render' if a.render else 'cpu_import'}
write_json(a.output,data)
if data['status']=='pass' and not a.render:write_json(base/'evidence/NATIVE_READY.json',data)
print(json.dumps(data,ensure_ascii=False))
raise SystemExit(0 if data['status']=='pass' else 4)
