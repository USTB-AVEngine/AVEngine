"""Exact-driver graphics DSOs, private extract-only installation and lease proof."""
import ctypes as C,hashlib,json,os,re,shutil,stat,subprocess,time,urllib.request
from pathlib import Path
from resources import write_json
GRAPHICS=("libEGL_nvidia","libGLX_nvidia","libnvidia-eglcore","libnvidia-glsi","libnvidia-glcore","libnvidia-glvkspirv","libnvidia-tls","libnvidia-gpucomp")
REQUIRED=("libEGL_nvidia","libnvidia-eglcore","libnvidia-glsi","libnvidia-glcore")
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def exact_version(kernel,smi):
 versions=set(re.findall(r'\b([0-9]{3}\.[0-9]+\.[0-9]+)\b',kernel))
 if len(versions)!=1 or versions!={smi.strip()}:raise ValueError('live kernel module and leased-GPU NVML driver versions must match exactly')
 return next(iter(versions))
def graphics_name(name,version):
 return any(name==stem+'.so.'+version for stem in GRAPHICS)
def live_version(gpu,out):
 kernel=Path('/proc/driver/nvidia/version').read_text();argv=['nvidia-smi','--id='+gpu,'--query-gpu=index,uuid,driver_version,pci.bus_id,mig.mode.current','--format=csv,noheader']
 p=subprocess.run(argv,capture_output=True,text=True,timeout=10);write_json(out/'driver_version_raw.json',{'kernel_raw':kernel,'argv':argv,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 if p.returncode:raise RuntimeError(p.stderr)
 rows=[line.split(',') for line in p.stdout.strip().splitlines()]
 if len(rows)!=1 or len(rows[0])!=5:raise ValueError('NVML selector is not one leased GPU')
 index,uuid,version,pci,mig=(v.strip() for v in rows[0]);version=exact_version(kernel,version)
 if index!=gpu or mig.lower()=='enabled':raise ValueError('exact full physical GPU lease required; MIG graphics excluded')
 return {'physical_gpu':gpu,'uuid':uuid,'version':version,'pci_bus_id':pci,'mig_mode':mig}
def device_files(gpu):
 result=[]
 for name in ['/dev/nvidiactl','/dev/nvidia'+gpu]:
  p=Path(name);s=p.stat()
  if not stat.S_ISCHR(s.st_mode) or not os.access(p,os.R_OK|os.W_OK):raise RuntimeError('existing leased device is not accessible: '+name)
  result.append({'path':name,'mode':oct(stat.S_IMODE(s.st_mode)),'uid':s.st_uid,'gid':s.st_gid,'rdev':s.st_rdev})
 return result
def download(url,path,limit,deadline):
 if path.exists():return {'reused':True,'bytes':path.stat().st_size}
 start=time.monotonic();total=0
 with urllib.request.urlopen(url,timeout=25) as response,path.open('xb') as f:
  if int(response.headers.get('Content-Length','0'))>limit:raise ValueError('package exceeds private budget')
  while block:=response.read(1024**2):
   total+=len(block)
   if total>limit or time.monotonic()>deadline:raise TimeoutError('bounded official package download exceeded')
   f.write(block)
  f.flush();os.fsync(f.fileno())
 return {'bytes':total,'seconds':time.monotonic()-start}
def prepare(base,driver,out,runtime_root=None):
 version=driver['version'];cache=base/'driver_cache'/version;cache.mkdir(parents=True,exist_ok=True)
 package='NVIDIA-Linux-x86_64-'+version+'-no-compat32.run';root='https://download.nvidia.com/XFree86/Linux-x86_64/'+version+'/'
 deadline=time.monotonic()+600;downloads={}
 for name,cap in [(package+'.sha256sum',65536),('license.txt',131072),(package,600*1024**2)]:downloads[name]=download(root+name,cache/name,cap,deadline)
 parts=(cache/(package+'.sha256sum')).read_text().split()
 if len(parts)!=2 or not re.fullmatch('[0-9a-f]{64}',parts[0]) or parts[1].lstrip('*')!=package or file_sha(cache/package)!=parts[0]:raise ValueError('official SHA256 mismatch; no extraction or DSO loading')
 runtime_parent=Path(runtime_root) if runtime_root else base/'driver_runtime'
 if not runtime_parent.resolve().is_relative_to(base.resolve()):raise ValueError('driver runtime outside owned base')
 runtime=runtime_parent/version;complete=runtime/'COMPLETE.json'
 if complete.exists():
  record=json.loads(complete.read_text())
  if record['version']!=version or any(file_sha(runtime/'lib'/r['filename'])!=r['sha256'] for r in record['libraries']):raise RuntimeError('private installed graphics library hash mismatch')
  write_json(out/'PRIVATE_DRIVER.json',record);return runtime
 runtime.mkdir(parents=True,exist_ok=False);quarantine=runtime/'extract_quarantine';quarantine.mkdir();private_tmp=runtime/'tmp';private_tmp.mkdir();env={**os.environ,'TMPDIR':str(private_tmp),'CUDA_VISIBLE_DEVICES':'','PATH':str(Path(__file__).parent/'driver_tar/bin')+':'+os.environ['PATH'],'CCI_NVIDIA_TAR_LOG':str(out/'driver_tar_raw.log')}
 argv=['sh',str(cache/package),'--extract-only','--target',str(quarantine/'payload')]
 from egl_diagnostic_entry import run_child
 # Separate verified child session; extraction never shares a worker process group.
 p=run_child(argv,env,out/'driver_extract_child.log',180)
 if p['status']!='passed':raise RuntimeError('official --extract-only failed; inspect driver_extract_child.log and driver_tar_raw.log; no installer was run')
 lib=runtime/'lib';lib.mkdir();libraries=[];seen=set()
 for p in (quarantine/'payload').rglob('*.so.*'):
  if not p.is_file() or p.is_symlink() or not graphics_name(p.name,version):continue
  # no-compat32 package contains only the 64-bit graphics DSOs; read ELF class explicitly.
  with p.open('rb') as f:header=f.read(5)
  if header!=b'\x7fELF\x02':raise ValueError('non-64-bit whitelisted graphics DSO')
  target=lib/p.name
  if target.exists():
   if file_sha(target)!=file_sha(p):raise ValueError('ambiguous duplicated graphics DSO')
   continue
  shutil.copyfile(p,target);target.chmod(0o644)
  elf=subprocess.run(['readelf','-d',str(target)],capture_output=True,text=True,timeout=10)
  if elf.returncode:raise RuntimeError('cannot inspect graphics SONAME/dependencies')
  sonames=re.findall(r'\(SONAME\).*?\[(.*?)\]',elf.stdout)
  if len(sonames)!=1 or '/' in sonames[0] or not any(sonames[0].startswith(s+'.so') for s in GRAPHICS):raise ValueError('unexpected graphics SONAME')
  link=lib/sonames[0]
  if link!=target:
   if link.exists() or link.is_symlink():raise ValueError('private SONAME collision')
   link.symlink_to(p.name)
  stem=p.name.split('.so.')[0];seen.add(stem);libraries.append({'filename':p.name,'bytes':target.stat().st_size,'sha256':file_sha(target),'SONAME':sonames[0],'ELF_dynamic_raw':elf.stdout})
 if not set(REQUIRED)<=seen:raise RuntimeError('official package lacks required EGL/GL graphics DSOs')
 vendor=runtime/'10_nvidia_private.json';write_json(vendor,{'file_format_version':'1.0.0','ICD':{'library_path':str(lib/'libEGL_nvidia.so.0')}})
 record={'version':version,'official_url':root+package,'package_sha256':parts[0],'downloads':downloads,'libraries':libraries,'vendor_json':str(vendor),'vendor_json_sha256':file_sha(vendor),'license_path':str(cache/'license.txt'),'license_sha256':file_sha(cache/'license.txt'),'license_title':'NVIDIA Driver License Agreement','installation':'--extract-only; copy whitelist of unchanged 64-bit EGL/GL user DSOs; installer/kernel tools never executed or exposed on PATH','runtime_cuda_nvml_kernel_driver_replaced':False,'redistributed':False,'private_quarantine':str(quarantine),'storage_note':'extracted package kept in personal quarantine for review; only graphics whitelist is on LD_LIBRARY_PATH'}
 write_json(complete,record);write_json(out/'PRIVATE_DRIVER.json',record);return runtime
class UUID(C.Structure):_fields_=[('bytes',C.c_ubyte*16)]
def prove_cuda_lease(driver):
 gpu=driver['physical_gpu']
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('CUDA visibility is not the single physical worker lease')
 cuda=C.CDLL('libcuda.so.1')
 def call(name,*args):
  result=getattr(cuda,name)(*args)
  if result:raise RuntimeError(name+' returned '+str(result))
 call('cuInit',0);count=C.c_int();call('cuDeviceGetCount',C.byref(count))
 if count.value!=1:raise ValueError('CUDA-visible device count must be exactly one')
 device=C.c_int();call('cuDeviceGet',C.byref(device),0);uuid=UUID();call('cuDeviceGetUuid',C.byref(uuid),device);raw=bytes(uuid.bytes).hex();u='GPU-'+raw[:8]+'-'+raw[8:12]+'-'+raw[12:16]+'-'+raw[16:20]+'-'+raw[20:];pci=C.create_string_buffer(64);call('cuDeviceGetPCIBusId',pci,64,device)
 if u.lower()!=driver['uuid'].lower():raise ValueError('CUDA ordinal UUID does not match physical NVML lease; refuse EGL context')
 return {'physical_gpu':gpu,'physical_uuid':driver['uuid'],'CUDA_VISIBLE_DEVICES':gpu,'cuda_ordinal':0,'cuda_device_handle':device.value,'cuda_uuid':u,'cuda_pci_bus_id':pci.value.decode(),'proof':'one visible CUDA device and exact UUID match; cuInit/enumeration only, no CUDA context or kernel was created'}
