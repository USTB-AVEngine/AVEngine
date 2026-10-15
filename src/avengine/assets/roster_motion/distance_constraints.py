"""Physical proposal margins, separate from unchanged QA answer predicates."""
import numpy as np

def matches_radial_window(values,sign,minimum_change_m=.5,maximum_retreat_m=.05):
    values=np.asarray(values,float)
    if len(values)<2 or not np.isfinite(values).all():return False
    progress=float(sign)*(values-values[0])
    return bool(progress[-1]>=float(minimum_change_m)-1e-9 and
        np.max(np.maximum.accumulate(progress)-progress)<=float(maximum_retreat_m)+1e-9)
