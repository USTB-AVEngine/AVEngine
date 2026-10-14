"""Fresh native process per RIR, isolating SDK state without changing physics."""
from pathlib import Path
from types import SimpleNamespace
import json,subprocess,sys,time
import numpy as np

class FreshProcessRenderer:
    def __init__(self,inputs,house,root,setup_report):
        self.inputs=inputs;self.house=house;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.index=0
        self.setup_report={**setup_report,'context_policy':{'lifetime':'new_process_context_and_scene_upload_for_every_RIR','temporal_coherence':False,'configured_thread_count':1,'no_seed_hook_or_binary_modification':True,'repeats':'independent deterministic native executions, not independent Monte Carlo seeds'}}
    def render(self,positions_m):
        arrays=[];cpu=0.;start=time.perf_counter()
        for position in positions_m:
            folder=self.root/f'{self.index:06d}';self.index+=1;folder.mkdir(exist_ok=False)
            request=folder/'request.json';wave=folder/'rir.npy';receipt=folder/'receipt.json'
            request.write_text(json.dumps({'inputs':self.inputs,'house':self.house,'position_m':list(position),'wave':str(wave),'receipt':str(receipt)},indent=2))
            with (folder/'native.log').open('xb') as log:
                subprocess.run([sys.executable,'-m','tools.assets.render_animal_rir_request','--request',str(request)],check=True,stdout=log,stderr=subprocess.STDOUT)
            evidence=json.loads(receipt.read_text());arrays.append(np.load(wave,allow_pickle=False));cpu+=evidence['process_cpu_seconds']
        return SimpleNamespace(samples=tuple(arrays),process_cpu_seconds=cpu,wall_seconds=time.perf_counter()-start)
