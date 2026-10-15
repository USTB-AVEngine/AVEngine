"""Construct actual physical sources using the same unchanged native DOA gate."""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np


class EnvironmentPositionAdmission:
    """Lazy one-thread CPU RIR probe; its cache is reused by actual UE admission."""
    def __init__(self,row,house,inputs):
        self.row=row;self.house=house;self.inputs=inputs;self.gate=None
        self.available=bool(house.get('acoustic_package_manifest') and inputs.get('simulation'))

    def check(self,endpoint,point):
        if not self.available:
            return {'status':'deferred_to_current_native_scene','unchanged_gate_deg':5.,
                    'reason':'geometry-only caller; actual native acoustic gate still mandatory'}
        if self.gate is None:
            from avengine.timeline.v7_position_admission import PositionAdmission
            row=self.row;layout=row['measured_static_layout'];house=deepcopy(self.house)
            room=deepcopy(next(r for r in house['checked_rooms']if r['room_label']==row['room_id']))
            room['placement']['camera_m']=list(layout['listener_m'])
            room['placement']['source_1_m']=list(layout['valid'][0]['position_m']);house['room']=room
            yaw=math.radians(layout['fixed_camera_yaw_deg'])
            scene={'listener_m':list(layout['listener_m']),'camera_forward':[math.sin(yaw),0.,-math.cos(yaw)],
                   'events':[{'endpoint_id':'environment'+str(i+1)}for i in range(len(row['construction_request']['required_sound_classes']))]}
            root=Path(house['input']['ue_config']['output_root'])/'native/parents'/row['scene_id']
            self.gate=PositionAdmission(scene,house,self.inputs,root/'rir_cache',root/'native_environment_position_prefilter.json')
        return self.gate.check(endpoint,np.asarray(point,float))

    def close(self):
        if self.gate is not None:self.gate.close()
