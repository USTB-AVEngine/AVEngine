"""Habitat orthographic RGB through a verified process-local llvmpipe context."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import ctypes,json,os,time,resource,traceback,subprocess
import numpy as np
import shapely
from shapely.geometry import shape
from PIL import Image
from .atlas import frame_polygon
from .pipeline import dump,now,load_native,overhead_for


def verify_software_devices():
    """Refuse before Simulator creation unless EGL sees exactly one software device."""
    required={'LIBGL_ALWAYS_SOFTWARE':'1','MESA_SHADER_CACHE_DISABLE':'true','LP_NUM_THREADS':'1'}
    if any(os.environ.get(k)!=v for k,v in required.items()):raise RuntimeError('Explicit process-local CPU rendering environment required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden for CPU rendering')
    if 'software_egl_only' not in os.environ.get('LD_PRELOAD',''):raise RuntimeError('Process-local software EGL filter required')
    lib=ctypes.CDLL(None);get=lib.eglGetProcAddress;get.argtypes=[ctypes.c_char_p];get.restype=ctypes.c_void_p
    query=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int))(get(b'eglQueryDevicesEXT'))
    string=ctypes.CFUNCTYPE(ctypes.c_char_p,ctypes.c_void_p,ctypes.c_int)(get(b'eglQueryDeviceStringEXT'))
    devices=(ctypes.c_void_p*8)();n=ctypes.c_int()
    if not query(8,devices,ctypes.byref(n)) or n.value!=1:raise RuntimeError('EGL software-only enumeration failed')
    extensions=string(devices[0],0x3055)
    if not extensions or b'EGL_MESA_device_software' not in extensions:raise RuntimeError('The sole EGL device is not software')
    return extensions.decode()


def cache_covers(row,floor,cache):
    """Geometry-only frame check, without inventing missing RGB pixels."""
    scope=shape(floor['floor_polygon']);frames=[];fy=floor['floor_y_m']
    for path in Path(cache).glob(row['house']+'__*.json'):
        entry=json.loads(path.read_text())
        if abs(entry['floor_y_m']-fy)>.3 or not path.with_suffix('.png').exists():continue
        im=next((x for x in entry['images'] if x['path'].endswith('_overview.png')),None)
        if im is None:continue
        try:frames.append(frame_polygon(entry,im,fy))
        except (ValueError,np.linalg.LinAlgError):continue
    if not frames:return False
    return scope.difference(shapely.union_all(frames)).area<=max(1e-8,scope.area*1e-6)


def render_house(job,output):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    started=time.time();hs=load_native();import magnum as mn
    extensions=verify_software_devices();scene=Path(job['scene_directory']);sid=scene.name.split('-',1)[1]
    cfg=hs.SimulatorConfiguration();cfg.scene_id=str(scene/(sid+'.glb'));cfg.gpu_device_id=-1;cfg.enable_physics=False
    sensor=hs.CameraSensorSpec();sensor.uuid='cpu_overhead';sensor.sensor_type=hs.SensorType.COLOR;sensor.sensor_subtype=hs.SensorSubType.ORTHOGRAPHIC;sensor.resolution=[1024,1024];sensor.ortho_scale=10
    sensor.position=mn.Vector3(0,0,0);sensor.orientation=mn.Vector3(-np.pi/2,np.pi,0);sensor.near=28.2;sensor.far=30.3
    agent=hs.agent.AgentConfiguration();agent.sensor_specifications=[sensor]
    sim=None;results=[]
    try:
        sim=hs.Simulator(hs.Configuration(cfg,[agent]))
        gl=ctypes.CDLL('libGL.so.1');gl.glGetString.argtypes=[ctypes.c_uint];gl.glGetString.restype=ctypes.c_char_p
        renderer=gl.glGetString(0x1F01);vendor=gl.glGetString(0x1F00)
        if not renderer or b'llvmpipe' not in renderer.lower():raise RuntimeError('GL_RENDERER is not verified llvmpipe; no observations authorized')
        native=sim._sensors['cpu_overhead']._sensor_object
        for target in job['targets']:
            row,floor=target['row'],target['floor'];key=row['house']+'__'+row['room_label']+'__'+floor['floor_id'];meta=Path(output)/(key+'.json');png=Path(output)/(key+'.png')
            if meta.exists() or png.exists():raise FileExistsError('Rendering output already exists: '+key)
            scope=shape(floor['floor_polygon']);x0,z0,x1,z1=scope.bounds;cx=(x0+x1)/2;cz=(z0+z1)/2;fy=floor['floor_y_m'];span=max(x1-x0,z1-z0,2)*1.10
            # Orthographic projection can be set directly on the live render camera.
            camera=native.render_camera;camera.projection_matrix=mn.Matrix4.orthographic_projection(mn.Vector2(span,span),28.2,30.3)
            state=hs.AgentState();state.position=mn.Vector3(cx,fy+30,cz);sim.initialize_agent(0,state)
            projection=np.asarray(camera.projection_matrix).tolist();view=np.asarray(camera.camera_matrix).tolist()
            entry={'house':row['house'],'source_region':row['room_label'],'floor_id':floor['floor_id'],'floor_y_m':fy,'orientation':'image_right=-X,image_up=+Z','view':view,'size_px':[1024,1024],'images':[{'path':str(png),'span_m':span,'projection':projection,'cut_height_above_floor_m':1.8}],'raw_glb':cfg.scene_id,'source':'Habitat CPU software orthographic rendering','compute_device':'CPU','gl_renderer':renderer.decode(),'gl_vendor':vendor.decode(),'egl_device_extensions':extensions,'near_m':28.2,'far_m':30.3,'camera_height_m':30,'floor_bottom_m':fy-.3,'software_device_filter':os.environ['LD_PRELOAD'],'created_at_utc':now(),'acoustics':'not_run'}
            frame=frame_polygon(entry,entry['images'][0],fy)
            if scope.difference(frame).area>max(1e-7,scope.area*1e-6):raise ValueError('Rendered camera frame does not cover source floor')
            clip=np.asarray(projection)@np.asarray(view);centre=clip@np.array([cx,fy,cz,1])
            if np.max(abs(centre[:2]))>1e-5:raise ValueError('Rendered view/projection centre inconsistent')
            image=sim.get_sensor_observations()['cpu_overhead'][:,:,:3]
            with png.open('xb') as f:Image.fromarray(image).save(f,format='PNG')
            dump(meta,entry);results.append({'id':key,'status':'rendered','floor_area_m2':floor['floor_area_m2'],'gl_renderer':renderer.decode()})
    except Exception as e:
        results.append({'house':job['house'],'status':'unresolved','error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc()})
    finally:
        if sim is not None:sim.close()
    return {'house':job['house'],'seconds':time.time()-started,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'results':results}


def run(root,workers=4):
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    root=Path(root);plan=json.loads((root/'processing_plan_v1.json').read_text());output=root/'overhead_cpu_render_v1';output.mkdir(exist_ok=False);jobs=[]
    for job in plan['jobs']:
        targets=[]
        for row in job['rows']:
            floors=row['floors'];largest=max(floors,key=lambda f:f['floor_area_m2']) if floors else None
            for floor in floors:
                if floor['floor_area_m2']<6 and floor is not largest:continue
                if not cache_covers(row,floor,plan['overhead_cache']):targets.append({'row':row,'floor':floor})
        if targets:jobs.append(dict(job,targets=targets))
    receipt={'started_at_utc':now(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'pid':os.getpid(),'machine':os.uname().nodename,'cpu_only':True,'workers':workers,'worker_memory_limit_gib':8,'parent_memory_limit_gib':6,'targets':sum(len(j['targets']) for j in jobs),'houses':len(jobs),'policy':'reuse existing same-floor orthographic RGB; render incomplete floor frames >=6 m² and each region largest floor; small discarded/unchanged layers without RGB remain explicitly unverified','jobs':[]}
    dump(root/'cpu_render_start_v1.json',receipt)
    with ProcessPoolExecutor(max_workers=workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(render_house,j,output):j['house'] for j in jobs}
        for future in as_completed(futures):
            try:result=future.result()
            except Exception as e:result={'house':futures[future],'results':[{'status':'unresolved','error':type(e).__name__+': '+str(e),'reason':'CPU_SOFTWARE_RENDER_WORKER_FAILED; no hardware fallback'}]}
            receipt['jobs'].append(result);print('RENDER_HOUSE',json.dumps(result,ensure_ascii=False),flush=True)
    receipt['finished_at_utc']=now();dump(root/'cpu_render_complete_v1.json',receipt)
    return receipt
