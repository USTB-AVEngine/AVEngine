"""Reuse an unchanged three-native-reference measurement at exactly one pose.

No position rounding, primary-kernel reads, RIR fitting or best-repeat selection.
Existing per-sample checkpoints always take priority over memoized measurements.
"""
import json,threading
import numpy as np

class ExactIndependentReferenceMemo:
    def __init__(self,identity):
        self.identity=json.dumps(identity,sort_keys=True,separators=(',',':'))
        self.values={};self.locks={};self.guard=threading.Lock()
    def get(self,position,measure,*,prefer_existing=False):
        point=np.asarray(position,dtype=np.float64)
        if point.shape!=(3,)or not np.isfinite(point).all():raise ValueError('finite exact native xyz required')
        key=(self.identity,point.tobytes())
        with self.guard:lock=self.locks.setdefault(key,threading.Lock())
        with lock:
            if prefer_existing:
                value=measure()
                if len(value['paths'])!=3:raise ValueError('three independent native references required')
                self.values.setdefault(key,value)
                return value,False
            if key in self.values:return self.values[key],True
            value=measure()
            if len(value['paths'])!=3:raise ValueError('three independent native references required')
            self.values[key]=value
            return value,False
