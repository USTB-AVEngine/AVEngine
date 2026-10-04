"""Reuse native one-thread RIRs to reject a placement before RGB production.

The frozen 5-degree direction predicate and RIR recipe are unchanged. A bad
sampled location is retained as a rejected location, never a qualified world.
"""
from pathlib import Path
from avengine.timeline.continuous_pilot import PositionRIRCache,write_json
from avengine.acoustics.house_acceptance_metrics import foa_direction

class PositionAdmission:
    def __init__(self,scene,house,inputs,cache_directory,receipt_path):
        from tools.timeline.pilot_continuous_scenes import native_renderer
        self.scene=scene;self.path=Path(receipt_path);self.rows=[]
        self.active={e['endpoint_id'] for e in scene['events']}
        renderer,identity=native_renderer(inputs,house)
        sim=identity['simulation']
        if sim['thread_count']!=1 or sim['diffraction'] is not True or sim['max_diffraction_order']!=10:
            raise ValueError('calibrated native RIR recipe required')
        self.cache=PositionRIRCache(cache_directory,renderer,identity)
    def check(self,endpoint,point):
        if endpoint not in self.active:return {'status':'not_applicable_silent_bystander'}
        rir=self.cache.get(point)
        direction=foa_direction(rir,self.scene['listener_m'],point,self.scene['camera_forward'],sample_rate=16000)
        error=direction['check']['angle_deg']
        row={'endpoint_id':endpoint,'actual_native_mouth_m':point.tolist(),'direction':direction,'unchanged_gate_deg':5.,'status':'pass' if error is not None and error<=5. else 'blocked'}
        self.rows.append(row);return row
    def close(self):
        write_json(self.path,{'status':'measured','checks':self.rows,'cache_stats':self.cache.stats,'unchanged_gate_deg':5.,'rejected_positions':sum(r['status']=='blocked' for r in self.rows)})
        self.cache.db.close()
