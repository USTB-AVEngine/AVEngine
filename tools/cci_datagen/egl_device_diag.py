#!/usr/bin/env python3
"""EGL client/device metadata, initialize only the exact leased physical device."""
import argparse,ctypes as C,ctypes.util,json,os,subprocess,socket
from pathlib import Path
EGL_EXTENSIONS=0x3055
def command(argv):
 try:
  p=subprocess.run(argv,capture_output=True,text=True,timeout=10)
  return {'argv':argv,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except Exception as e:return {'argv':argv,'error':str(e)}
def metadata():
 roots=[Path('/usr/lib/x86_64-linux-gnu'),Path('/lib/x86_64-linux-gnu'),Path('/usr/lib64'),Path('/usr/local/nvidia/lib64'),Path('/usr/local/nvidia/lib')]
 roots.extend(Path(s) for s in os.environ.get('LD_LIBRARY_PATH','').split(':') if s)
 libraries={}
 for root in roots:
  if root.is_dir():
   for pattern in ['libEGL*','libGLX*nvidia*','libnvidia*egl*','libOSMesa*']:
    for p in root.glob(pattern):libraries[str(p)]={'target':str(p.resolve()),'exists':p.exists()}
 vendors=[]
 for root in [Path('/usr/share/glvnd/egl_vendor.d'),Path('/etc/glvnd/egl_vendor.d')]:
  if root.exists():
   for p in root.glob('*.json'):vendors.append({'path':str(p),'text':p.read_text()})
 return {'host':socket.gethostname(),'environment':{k:os.environ.get(k) for k in ['NVIDIA_DRIVER_CAPABILITIES','NVIDIA_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','EGL_PLATFORM','__EGL_VENDOR_LIBRARY_FILENAMES','LD_LIBRARY_PATH']},'libraries':libraries,'vendors':vendors,'nvidia_list':command(['nvidia-smi','-L']),'driver':command(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,pci.bus_id,mig.mode.current','--format=csv,noheader']),'ldconfig':command(['ldconfig','-p'])}
def selected_device(devices,gpu):
 if not str(gpu).isdigit():return None
 matches=[d for d in devices if d.get('cuda_device')==int(gpu)]
 return matches[0] if len(matches)==1 else None
def egl_query(library,gpu,*,initialize=False,mig=False,expected_uuid=None):
 result={'library':library,'assigned_gpu':gpu,'MIG':mig,'hardware_context_created':False}
 try:egl=C.CDLL(library)
 except OSError as e:return {**result,'status':'library_unavailable','error':str(e)}
 egl.eglQueryString.argtypes=[C.c_void_p,C.c_int];egl.eglQueryString.restype=C.c_char_p
 egl.eglGetError.restype=C.c_uint;egl.eglGetProcAddress.argtypes=[C.c_char_p];egl.eglGetProcAddress.restype=C.c_void_p
 def error():return hex(egl.eglGetError())
 def extension(name,restype,*args):
  ptr=egl.eglGetProcAddress(name.encode())
  return C.CFUNCTYPE(restype,*args)(ptr) if ptr else None
 raw=egl.eglQueryString(None,EGL_EXTENSIONS);client=raw.decode(errors='replace') if raw else ''
 result.update(client_extensions=client,client_query_error=error())
 query=extension('eglQueryDevicesEXT',C.c_uint,C.c_int,C.POINTER(C.c_void_p),C.POINTER(C.c_int))
 strings=extension('eglQueryDeviceStringEXT',C.c_char_p,C.c_void_p,C.c_int)
 attributes=extension('eglQueryDeviceAttribEXT',C.c_uint,C.c_void_p,C.c_int,C.POINTER(C.c_ssize_t))
 binary=extension('eglQueryDeviceBinaryEXT',C.c_uint,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.POINTER(C.c_int))
 count=C.c_int();handles=(C.c_void_p*64)();devices=[]
 if query and 'EGL_EXT_device_enumeration' in client:
  ok=query(64,handles,C.byref(count));result.update(enumeration_return=bool(ok),enumeration_error=error(),device_count=count.value)
  for i in range(count.value if ok else 0):
   ext=strings(handles[i],EGL_EXTENSIONS) if strings else None
   d={'egl_index':i,'extensions':ext.decode(errors='replace') if ext else ''}
   if 'EGL_NV_device_cuda' in d['extensions'] and attributes:
    cuda=C.c_ssize_t(-1);valid=attributes(handles[i],0x323A,C.byref(cuda));d.update(cuda_device=int(cuda.value) if valid else None,cuda_query_error=error())
   if 'EGL_EXT_device_drm' in d['extensions'] and strings:
    value=strings(handles[i],0x3233);d['drm_device_file']=value.decode(errors='replace') if value else None
   if 'EGL_EXT_device_persistent_id' in d['extensions'] and binary:
    value=(C.c_ubyte*16)();size=C.c_int();ok=binary(handles[i],0x335C,16,value,C.byref(size));d['device_uuid_hex']=bytes(value).hex() if ok and size.value==16 else None;d['uuid_query_error']=error()
   devices.append(d)
 else:result.update(device_count=0,enumeration_note='client does not advertise EGL_EXT_device_enumeration')
 result['devices']=devices
 egl.eglGetDisplay.argtypes=[C.c_void_p];egl.eglGetDisplay.restype=C.c_void_p
 default=egl.eglGetDisplay(None);result.update(default_display_handle=hex(default or 0),default_display_error=error(),default_display_initialized=False)
 selected=selected_device(devices,gpu);result['selected_leased_device']=selected
 if selected and expected_uuid:
  wanted=expected_uuid.removeprefix('GPU-').replace('-','').lower();actual=selected.get('device_uuid_hex')
  result['physical_uuid_proof']={'expected':expected_uuid,'EGL_uuid_hex':actual,'CUDA_ordinal':gpu,'proof':'EGL_NV_device_cuda compatible CUDA handle is separately UUID-verified; EGL persistent UUID also checked when available'}
  if actual and actual.lower()!=wanted:
   result.update(status='blocked_physical_UUID_mismatch',device_display_initialized=False);return result
 if initialize and selected and not mig:
  get_display=extension('eglGetPlatformDisplayEXT',C.c_void_p,C.c_uint,C.c_void_p,C.POINTER(C.c_int))
  if get_display:
   display=get_display(0x313F,handles[selected['egl_index']],None);result.update(device_display_handle=hex(display or 0),device_display_error=error())
   egl.eglInitialize.argtypes=[C.c_void_p,C.POINTER(C.c_int),C.POINTER(C.c_int)];egl.eglInitialize.restype=C.c_uint
   major=C.c_int();minor=C.c_int();ok=egl.eglInitialize(display,C.byref(major),C.byref(minor)) if display else 0
   result.update(device_display_initialized=bool(ok),initialize_error=error(),egl_version=[major.value,minor.value])
   if ok:
    raw=egl.eglQueryString(display,0x3053);result['display_vendor']=raw.decode(errors='replace') if raw else None
    egl.eglTerminate.argtypes=[C.c_void_p];egl.eglTerminate(display)
 else:result['device_initialize_note']='MIG graphics excluded or no unique CUDA-ID match; no other physical device initialized'
 result['status']='metadata_complete';return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--library',default='libEGL.so.1');p.add_argument('--gpu',required=True);p.add_argument('--initialize-leased-device',action='store_true');p.add_argument('--mig',action='store_true');p.add_argument('--expected-uuid');a=p.parse_args()
 print(json.dumps(egl_query(a.library,a.gpu,initialize=a.initialize_leased_device,mig=a.mig,expected_uuid=a.expected_uuid)))
if __name__=='__main__':main()
